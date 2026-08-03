function stats = WadaDiagnostic(M, varargin)
%WADADIAGNOSTIC  Numerical signatures of "Fatou set = d attracting basins
%   with common (Wada) boundary", per Fisher-Hill-Lazebnik-Thompson.
%
%   stats = WadaDiagnostic(M)
%   stats = WadaDiagnostic(M, 'Radius', r)
%
%   M        basin label matrix from BasinPlot (0 = unresolved,
%            1..d = basin index).
%
%   Returns a struct:
%     stats.nBasins            number of distinct nonzero labels present
%     stats.unresolvedFraction fraction of ALL pixels labeled 0. For maps
%                              of the paper's type (Fatou set is EXACTLY a
%                              disjoint union of attracting basins), this
%                              should tend to 0 as MaxIter and Resolution
%                              grow -- the Julia set has measure zero for
%                              these maps' typical cases, and there are no
%                              rotation domains to get stuck in. A
%                              persistent unresolved REGION (not just
%                              boundary dust) falsifies the candidate.
%     stats.boundaryFraction   fraction of pixels that are boundary pixels
%                              (>= 2 distinct nonzero labels within
%                              Radius).
%     stats.wadaFraction       among boundary pixels, the fraction whose
%                              Radius-neighborhood contains ALL nBasins
%                              labels. For a genuine Wada configuration
%                              this tends to 1 as Resolution increases;
%                              for a non-Wada map (e.g. two basins meeting
%                              along an arc away from the others) it
%                              plateaus well below 1.
%
%   IMPORTANT CAVEAT on wadaFraction at finite resolution: the true Wada
%   property says every boundary point has all d basins in EVERY
%   neighborhood -- an infinite-resolution statement. At pixel scale, a
%   radius-1 neighborhood of a boundary pixel often shows only the two
%   locally-dominant basins even for genuinely-Wada maps, because the
%   third basin's presence is at a finer scale than one pixel. So treat
%   wadaFraction as a RESOLUTION-DEPENDENT diagnostic: compute it at
%   increasing Resolution (and/or increasing Radius) and look at the
%   trend, not the absolute number at one setting.
%
%   Name-Value options
%   -------------------
%     Radius   half-width (in pixels) of the square neighborhood used
%              for both boundary detection and the Wada test (default 2)

    p = inputParser;
    addRequired(p, 'M', @(v) isnumeric(v) && ismatrix(v));
    addParameter(p, 'Radius', 2, @(v) isnumeric(v) && isscalar(v) && v >= 1);
    parse(p, M, varargin{:});
    r = round(p.Results.Radius);

    labels = unique(M(M > 0));
    d = numel(labels);

    stats = struct();
    stats.nBasins = d;
    stats.unresolvedFraction = mean(M(:) == 0);

    if d < 2
        stats.boundaryFraction = 0;
        stats.wadaFraction = NaN;
        return
    end

    % ---- per-label dilation: which pixels have label L within Radius --------
    % done with a box dilation via cumulative moving max on a binary mask;
    % imdilate would be simpler but requires the Image Processing Toolbox,
    % which this project doesn't otherwise need.
    [nRows, nCols] = size(M);
    nearLabel = false(nRows, nCols, d);
    for k = 1:d
        B = (M == labels(k));
        nearLabel(:,:,k) = boxDilate(B, r);
    end

    labelsNearby = sum(nearLabel, 3);

    isBoundary = labelsNearby >= 2;
    isWada     = labelsNearby == d;

    stats.boundaryFraction = mean(isBoundary(:));
    nBoundary = sum(isBoundary(:));
    if nBoundary == 0
        stats.wadaFraction = NaN;
    else
        stats.wadaFraction = sum(isWada(:) & isBoundary(:)) / nBoundary;
    end
end

% =========================================================================
function D = boxDilate(B, r)
%BOXDILATE  Binary dilation by a (2r+1)x(2r+1) box, toolbox-free.
    D = B;
    % dilate rows
    for s = 1:r
        D(1:end-s, :) = D(1:end-s, :) | B(1+s:end, :);
        D(1+s:end, :) = D(1+s:end, :) | B(1:end-s, :);
    end
    B2 = D;
    % dilate columns
    for s = 1:r
        D(:, 1:end-s) = D(:, 1:end-s) | B2(:, 1+s:end);
        D(:, 1+s:end) = D(:, 1+s:end) | B2(:, 1:end-s);
    end
end