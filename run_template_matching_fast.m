function run_template_matching()
% RUN_TEMPLATE_MATCHING  Multi-scale rotation-invariant template matching
%   using CFOG pixel-wise features and masked ZNCC via FFT on GPU.
%
%   This script performs a coarse grid search over discrete scales and
%   rotation angles to locate a small template image within a large
%   reference image. The matching score is the zero-mean normalised
%   cross-correlation (ZNCC) computed in the frequency domain with a
%   validity mask that accounts for boundary effects introduced by
%   rotation. Channel-pair complex packing is used to halve the number
%   of FFT operations on the multi-channel CFOG descriptor.

    clear; clc;

    addpath('.\pixel-wise feature represatation\CFOG');

    % ======================== Parameters ========================
    resize_factor       = 0.2;        % down-sample ratio for speed
    angle_step          = 10;           % rotation step in degrees
    search_scales       = 0.6:0.05:1;  % template scale range

    eps_denom           = 1e-6;        % regulariser for ZNCC denominator
    min_valid_ratio     = 0.55;        % skip rotations with too much padding
    % ============================================================

    % Load images
    try
        im_ref  = imread('.\Registration_test\map.jpg');
        im_tpl  = imread('.\Registration_test\2.jpg');
    catch
        error('Cannot find images. Please check the file paths.');
    end

    if size(im_ref, 3) == 3, im_ref = rgb2gray(im_ref); end
    if size(im_tpl, 3) == 3, im_tpl = rgb2gray(im_tpl); end
    im_ref = double(im_ref);
    im_tpl = double(im_tpl);

    % Initialise GPU
    gpuDev = gpuDevice;
    fprintf('GPU: %s (Memory: %.1f GB)\n', gpuDev.Name, gpuDev.TotalMemory / 1e9);

    %% ==================== Template Matching ====================
    fprintf('\n=== Multi-Scale Rotational Template Matching ===\n');

    im_ref_ds  = imresize(im_ref, resize_factor);
    im_tpl_ds  = imresize(im_tpl, resize_factor);

    % Extract CFOG features for the reference image (pre-computation,
    % excluded from the matching timer).
    fprintf('Extracting CFOG features for reference image ...\n');
    [feat_ref, flag] = pixelFeature(im_ref_ds, 'CFOG');
    if flag == 0, error('CFOG feature extraction failed for reference image.'); end

    % ===== Start matching timer =====
    wait(gpuDev);
    t_start = tic;

    % Pre-compute FFT cache for the reference feature map.
    [fft_cache, ok] = build_reference_fft_cache(feat_ref, im_tpl_ds, search_scales);
    if ~ok, error('Failed to build reference FFT cache.'); end

    best_score = -inf;
    best_angle = 0;
    best_scale = 1;
    best_index = 1;

    angles      = 0 : angle_step : 350;
    total_steps = numel(search_scales) * numel(angles);
    counter     = 0;

    for s = search_scales
        im_tpl_scaled = imresize(im_tpl_ds, s);

        for ang = angles
            [im_tpl_rot, mask_rot] = rotate_with_mask(im_tpl_scaled, ang);

            % Skip configurations where the valid region is too small.
            if mean(mask_rot(:)) < min_valid_ratio
                counter = counter + 1;
                continue;
            end

            [feat_tpl, ~] = pixelFeature(im_tpl_rot, 'CFOG');

            % Compute ZNCC via FFT; only the maximum value and its linear
            % index are transferred back from the GPU.
            [score, idx] = compute_zncc_max(fft_cache, feat_tpl, mask_rot, eps_denom);

            if score > best_score
                best_score = score;
                best_angle = ang;
                best_scale = s;
                best_index = idx;
            end

            counter = counter + 1;
            if mod(counter, 20) == 0
                fprintf('Progress: %5.1f%%  (Scale %.2f, Angle %3d°)  Best score: %.4f\n', ...
                    (counter / total_steps) * 100, s, ang, best_score);
            end
        end
    end

    wait(gpuDev);
    elapsed = toc(t_start);
    % ===== End matching timer =====

    fprintf('\nMatching finished: Scale = %.2f, Angle = %d°, Score = %.4f, Time = %.2f s\n', ...
        best_scale, best_angle, best_score, elapsed);

    % Convert the best linear index back to (row, col) in the down-sampled
    % reference image, then map to full-resolution coordinates.
    [r_ds, c_ds] = ind2sub([fft_cache.H_ref, fft_cache.W_ref], best_index);
    centre_xy = [c_ds, r_ds] / resize_factor;

    % Release GPU memory occupied by the FFT cache.
    clear fft_cache feat_ref;

    %% ====================== Visualisation ======================
    fprintf('\n=========================================\n');
    fprintf('  Final Result\n');
    fprintf('=========================================\n');
    fprintf('  Scale  : %.2f\n', best_scale);
    fprintf('  Angle  : %d°\n',  best_angle);
    fprintf('  Score  : %.4f   (range [-1, 1])\n', best_score);
    fprintf('  Centre : (%.1f, %.1f) px\n', centre_xy(1), centre_xy(2));
    fprintf('  Time   : %.2f s\n', elapsed);
    fprintf('-----------------------------------------\n');

    figure;
    imshow(uint8(im_ref)); hold on;
    title(sprintf('Angle %d°, Scale %.2f, Score %.3f  (%.2f s)', ...
        best_angle, best_scale, best_score, elapsed));

    cx = centre_xy(1);
    cy = centre_xy(2);
    plot(cx, cy, 'r+', 'MarkerSize', 10, 'LineWidth', 2);

    % Draw the oriented bounding box of the detected template.
    h_tpl = size(im_tpl, 1) * best_scale;
    w_tpl = size(im_tpl, 2) * best_scale;

    theta = deg2rad(-best_angle);
    corners = [-w_tpl/2, -h_tpl/2;
                w_tpl/2, -h_tpl/2;
                w_tpl/2,  h_tpl/2;
               -w_tpl/2,  h_tpl/2];
    R = [cos(theta), -sin(theta);
         sin(theta),  cos(theta)];
    corners_rot = (R * corners')';

    xc = corners_rot(:,1) + cx;
    yc = corners_rot(:,2) + cy;
    xc = [xc; xc(1)];
    yc = [yc; yc(1)];
    plot(xc, yc, 'r-', 'LineWidth', 2);

    text(cx, cy - h_tpl/2 - 10, sprintf('Score: %.2f', best_score), ...
        'Color', 'r', 'FontWeight', 'bold');
end


%% ====================================================================
%  rotate_with_mask — Rotate an image and produce a binary validity mask
% =====================================================================
function [im_rot, mask_rot] = rotate_with_mask(im_in, angle_deg)
% ROTATE_WITH_MASK  Rotate an image by a given angle (loose bounding box)
%   and return the rotated image together with a binary mask indicating
%   which pixels contain valid (non-padded) data.
%
%   im_in     — input image (2-D, double)
%   angle_deg — counter-clockwise rotation in degrees
%
%   im_rot    — rotated image  (bilinear interpolation, loose crop)
%   mask_rot  — binary mask (1 = valid, 0 = padding)

    im_in = double(im_in);
    mask  = ones(size(im_in), 'double');

    im_rot   = imrotate(im_in, angle_deg, 'bilinear', 'loose');
    mask_rot = imrotate(mask,  angle_deg, 'nearest',  'loose');

    mask_rot = double(mask_rot > 0.5);
end


%% ====================================================================
%  build_reference_fft_cache — Pre-compute packed FFTs for the reference
% =====================================================================
function [cache, ok] = build_reference_fft_cache(feat_ref, tpl_base, search_scales)
% BUILD_REFERENCE_FFT_CACHE  Pre-compute the frequency-domain
%   representation of the multi-channel CFOG reference feature map.
%
%   Adjacent channel pairs are packed into a single complex FFT to halve
%   the total number of transforms.  The FFT size is chosen to accommodate
%   the largest possible template (worst-case scale + loose rotation) and
%   is rounded up to the nearest 5-smooth number for efficient FFT.
%
%   feat_ref      — H×W×D reference feature volume (double)
%   tpl_base      — down-sampled template image used to estimate sizes
%   search_scales — vector of template scale factors
%
%   cache — struct with pre-computed FFTs and metadata
%   ok    — true if construction succeeded

    ok    = false;
    cache = struct();

    [H1, W1, D] = size(feat_ref);
    if D < 1, return; end

    % Determine the largest template bounding box across all scales and a
    % full 360° rotation (diagonal of the scaled template).
    max_tpl_h = 0;
    max_tpl_w = 0;
    for s = search_scales
        tmp     = imresize(tpl_base, s);
        diaglen = ceil(sqrt(double(size(tmp,1))^2 + double(size(tmp,2))^2));
        max_tpl_h = max(max_tpl_h, diaglen);
        max_tpl_w = max(max_tpl_w, diaglen);
    end

    fftH = next_smooth_number(H1 + max_tpl_h - 1);
    fftW = next_smooth_number(W1 + max_tpl_w - 1);

    cache.H_ref = H1;
    cache.W_ref = W1;
    cache.D     = D;
    cache.fftH  = fftH;
    cache.fftW  = fftW;

    % Per-pixel sums over channels (needed for ZNCC mean / variance).
    I_sum  = sum(feat_ref,    3);
    I2_sum = sum(feat_ref.^2, 3);

    cache.F_I_sum  = fft2(gpuArray(I_sum),  fftH, fftW);
    cache.F_I2_sum = fft2(gpuArray(I2_sum), fftH, fftW);

    % Packed channel-pair FFTs: channels (2k-1, 2k) are stored as
    % real + j·imag in a single complex FFT.
    num_packs       = ceil(D / 2);
    cache.F_packed  = cell(1, num_packs);

    k = 1;
    d = 1;
    while d <= D
        if d < D
            I_pack = gpuArray(feat_ref(:,:,d)) + 1i * gpuArray(feat_ref(:,:,d+1));
            cache.F_packed{k} = fft2(I_pack, fftH, fftW);
            d = d + 2;
        else
            cache.F_packed{k} = fft2(gpuArray(feat_ref(:,:,d)), fftH, fftW);
            d = d + 1;
        end
        k = k + 1;
    end

    ok = true;
end


%% ====================================================================
%  compute_zncc_max — Masked ZNCC via FFT (packed), return max only
% =====================================================================
function [max_val, max_idx] = compute_zncc_max(cache, feat_tpl, mask_tpl, eps_denom)
% COMPUTE_ZNCC_MAX  Compute the masked zero-mean normalised cross-
%   correlation (ZNCC) between the cached reference feature map and a
%   rotated/scaled template feature map, using FFT-based convolution.
%
%   Only the global maximum of the score map and its linear index are
%   transferred from the GPU, keeping data-transfer overhead minimal.
%
%   cache    — struct returned by build_reference_fft_cache
%   feat_tpl — H2×W2×D template CFOG feature volume
%   mask_tpl — H2×W2 binary mask (1 = valid pixel)
%   eps_denom— small constant to regularise the denominator
%
%   max_val  — scalar, best ZNCC score in [-1, 1]
%   max_idx  — linear index of the best score in the H_ref×W_ref map

    [H2, W2, Dt] = size(feat_tpl);
    D = cache.D;

    if Dt ~= D
        error('Channel mismatch: reference has %d channels, template has %d.', D, Dt);
    end
    if any(size(mask_tpl) ~= [H2, W2])
        error('mask_tpl size must match the spatial dimensions of feat_tpl.');
    end

    H_ref = cache.H_ref;
    W_ref = cache.W_ref;
    fftH  = cache.fftH;
    fftW  = cache.fftW;

    w_count = sum(mask_tpl(:));          % number of valid template pixels
    if w_count < 1
        max_val = -inf;
        max_idx = 1;
        return;
    end

    % ---- Mask correlation (GPU) ----
    mask_flip = rot90(mask_tpl, 2);
    F_mask    = fft2(gpuArray(mask_flip), fftH, fftW);

    sum_Iw_full  = real(ifft2(cache.F_I_sum  .* F_mask));
    sum_I2w_full = real(ifft2(cache.F_I2_sum .* F_mask));

    [sum_Iw, sum_I2w] = crop_valid_region(sum_Iw_full, sum_I2w_full, ...
                                           H_ref, W_ref, H2, W2);

    % ---- Template-side statistics (CPU scalars) ----
    mask3  = reshape(mask_tpl, [H2, W2, 1]);
    Tw     = sum(feat_tpl .* mask3, 'all');
    T2w    = sum((feat_tpl.^2) .* mask3, 'all');

    n_eff  = w_count * D;                % effective sample count
    mu_T   = Tw / n_eff;

    var_T  = T2w - 2 * mu_T * Tw + (mu_T^2) * n_eff;
    if var_T < 0, var_T = 0; end
    sigma_T = sqrt(var_T);

    % ---- Cross-correlation via packed FFTs (GPU) ----
    acc_freq = [];
    k = 1;
    d = 1;

    while d <= D
        if d < D
            % Pack conjugate pair: T_even - j·T_odd  (matched to I_even + j·I_odd)
            T_pack = (feat_tpl(:,:,d) .* mask_tpl) ...
                - 1i * (feat_tpl(:,:,d+1) .* mask_tpl);
            F_T    = fft2(gpuArray(rot90(T_pack, 2)), fftH, fftW);
            term   = cache.F_packed{k} .* F_T;
            d = d + 2;
        else
            T_single = feat_tpl(:,:,d) .* mask_tpl;
            F_T      = fft2(gpuArray(rot90(T_single, 2)), fftH, fftW);
            term     = cache.F_packed{k} .* F_T;
            d = d + 1;
        end

        if isempty(acc_freq)
            acc_freq = term;
        else
            acc_freq = acc_freq + term;
        end
        k = k + 1;
    end

    sum_ITw_full = real(ifft2(acc_freq));
    sum_ITw      = crop_valid_region(sum_ITw_full, [], H_ref, W_ref, H2, W2);

    % ---- ZNCC score map (GPU, element-wise) ----
    mu_I = sum_Iw / n_eff;

    var_I = sum_I2w - 2 .* mu_I .* sum_Iw + (mu_I.^2) .* n_eff;
    var_I(var_I < 0) = 0;
    sigma_I = sqrt(var_I);

    numerator   = sum_ITw ...
                - mu_T .* sum_Iw ...
                - mu_I .* Tw ...
                + (mu_I .* mu_T) .* n_eff;

    denominator = (sigma_I .* sigma_T) + eps_denom;

    score_map = numerator ./ denominator;
    score_map(score_map >  1) =  1;
    score_map(score_map < -1) = -1;

    % Return only the global maximum (scalar transfer from GPU).
    [max_val_gpu, max_idx_gpu] = max(score_map(:));
    max_val = gather(max_val_gpu);
    max_idx = gather(max_idx_gpu);
end


%% ====================================================================
%  crop_valid_region — Extract the 'same'-size centre from FFT output
% =====================================================================
function [A_crop, B_crop] = crop_valid_region(A_full, B_full, H1, W1, H2, W2)
% CROP_VALID_REGION  Extract the central H1×W1 block from the full
%   (H1+H2-1)×(W1+W2-1) correlation output, equivalent to MATLAB's
%   'same' option anchored on the first operand.

    r0 = floor(H2/2) + 1;
    c0 = floor(W2/2) + 1;

    A_crop = A_full(r0:r0+H1-1, c0:c0+W1-1);
    if ~isempty(B_full)
        B_crop = B_full(r0:r0+H1-1, c0:c0+W1-1);
    else
        B_crop = [];
    end
end


%% ====================================================================
%  next_smooth_number — Smallest 5-smooth integer >= n
% =====================================================================
function n = next_smooth_number(n)
% NEXT_SMOOTH_NUMBER  Return the smallest integer >= n whose prime
%   factors are limited to {2, 3, 5} (a.k.a. regular / 5-smooth numbers).
%   Such sizes yield the most efficient radix-based FFT execution.

    while true
        m = n;
        while mod(m, 2) == 0, m = m / 2; end
        while mod(m, 3) == 0, m = m / 3; end
        while mod(m, 5) == 0, m = m / 5; end
        if m == 1
            return;
        end
        n = n + 1;
    end
end