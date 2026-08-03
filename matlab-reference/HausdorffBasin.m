function [dChord, dEucl, details] = HausdorffBasin(M, x, y, targetBoundary, varargin)
%HAUSDORFFBASIN  Hausdorff distance between the BasinPlot Julia set and a
%   target boundary curve, reported in BOTH the chordal (spherical) and
%   Euclidean metrics.
%
%   [dChord, dEucl, details] = HausdorffBasin(M, x, y, targetBoundary)
%
%   INPUTS
%     M               basin-label matrix from BasinPlot (0=unresolved,
%                     k=basin k). Its Julia set is extracted as the set of
%                     boundary pixels: pixels adjacent to a different
%                     nonzero label.
%     x, y            the axis vectors BasinPlot returned (real/imag of the
%                     grid), used to convert pixel indices to complex
%                     coordinates.
%     targetBoundary  complex vector of points sampled from the TARGET
%                     boundary (e.g. the image's region boundaries), in the
%                     SAME coordinate plane as x,y.
%
%   OUTPUTS
%     dChord   symmetric Hausdorff distance in the chordal (spherical)
%              metric d(z,w)=2|z-w|/sqrt((1+|z|^2)(1+|w|^2)). This is the
%              metric the paper (Notation 1.1) measures approximation in,
%              so it is the number to quote against Theorem A's epsilon.
%     dEucl    symmetric Hausdorff distance in the ordinary Euclidean
%              metric on the plane -- easier to interpret as "pixels off"
%              but not sphere-aware (penalizes near-infinity mismatch
%              differently).
%     details  struct with directed distances and the extracted Julia set:
%                .juliaPoints      complex points of the computed Julia set
%                .dChord_J2T .dChord_T2J   directed chordal distances
%                .dEucl_J2T  .dEucl_T2J    directed Euclidean distances
%              (symmetric Hausdorff = max of the two directed distances.)
%
%   Name-Value
%     MaxPoints   subsample each point set to at most this many points
%                 before the O(nm) distance (keeps it fast at high res).
%                 (default 4000)
%
%   NOTE on directedness: Hausdorff = max(sup_{a in A} inf_{b in B} d,
%   sup_{b in B} inf_{a in A} d). The two directed pieces answer different
%   questions -- dChord_J2T large means the Julia set has stray pieces far
%   from the target (spurious structure); dChord_T2J large means part of
%   the target boundary has no nearby Julia set (missed boundary). The
%   details struct exposes both so a bad number tells you WHICH failure.

    p = inputParser;
    addRequired(p, 'M');
    addRequired(p, 'x');
    addRequired(p, 'y');
    addRequired(p, 'targetBoundary');
    addParameter(p, 'MaxPoints', 4000, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, M, x, y, targetBoundary, varargin{:});
    maxPts = round(p.Results.MaxPoints);

    % ---- extract the computed Julia set as basin-boundary pixels --------
    lab = M;
    isB = false(size(lab));
    isB(1:end-1,:) = isB(1:end-1,:) | (lab(1:end-1,:) ~= lab(2:end,:) & lab(1:end-1,:) > 0 & lab(2:end,:) > 0);
    isB(2:end,:)   = isB(2:end,:)   | (lab(2:end,:)   ~= lab(1:end-1,:) & lab(2:end,:) > 0 & lab(1:end-1,:) > 0);
    isB(:,1:end-1) = isB(:,1:end-1) | (lab(:,1:end-1) ~= lab(:,2:end) & lab(:,1:end-1) > 0 & lab(:,2:end) > 0);
    isB(:,2:end)   = isB(:,2:end)   | (lab(:,2:end)   ~= lab(:,1:end-1) & lab(:,2:end) > 0 & lab(:,1:end-1) > 0);

    [iy, ix] = find(isB);
    juliaPts = x(ix).' + 1i*y(iy).';

    tgt = targetBoundary(:);

    juliaPts = subsample(juliaPts, maxPts);
    tgt      = subsample(tgt, maxPts);

    if isempty(juliaPts)
        warning('HausdorffBasin:emptyJulia', 'No basin-boundary pixels found; is M single-basin?');
        dChord = Inf; dEucl = Inf; details = struct(); return
    end

    % ---- directed distances (both metrics) ------------------------------
    [dChord_J2T, dEucl_J2T] = directed(juliaPts, tgt);
    [dChord_T2J, dEucl_T2J] = directed(tgt, juliaPts);

    dChord = max(dChord_J2T, dChord_T2J);
    dEucl  = max(dEucl_J2T,  dEucl_T2J);

    details = struct('juliaPoints', juliaPts, ...
        'dChord_J2T', dChord_J2T, 'dChord_T2J', dChord_T2J, ...
        'dEucl_J2T',  dEucl_J2T,  'dEucl_T2J',  dEucl_T2J);

    fprintf('Hausdorff (Julia vs target):\n');
    fprintf('   chordal:   %.4e   [J->T %.4e, T->J %.4e]\n', dChord, dChord_J2T, dChord_T2J);
    fprintf('   euclidean: %.4e   [J->T %.4e, T->J %.4e]\n', dEucl,  dEucl_J2T,  dEucl_T2J);
end

% =========================================================================
function [dc, de] = directed(A, B)
% directed Hausdorff sup_{a in A} inf_{b in B} d(a,b), both metrics.
% Chunked to bound memory: for each a, min over all b.
    nA = numel(A);
    dc = 0; de = 0;
    chunk = 500;
    for i0 = 1:chunk:nA
        idx = i0:min(i0+chunk-1, nA);
        a = A(idx);                      % col
        diff = a(:) - B(:).';            % |idx| x |B|
        eu = abs(diff);
        [minEu, j] = min(eu, [], 2);
        de = max(de, max(minEu));
        % chordal for the same nearest-Euclidean partner is not necessarily
        % the chordal-nearest; compute chordal min directly:
        ch = 2*abs(diff) ./ sqrt((1+abs(a(:)).^2) .* (1+abs(B(:).').^2));
        minCh = min(ch, [], 2);
        dc = max(dc, max(minCh));
    end
end

% =========================================================================
function v = subsample(v, maxPts)
    if numel(v) > maxPts
        idx = round(linspace(1, numel(v), maxPts));
        v = v(idx);
    end
    v = v(:);
end