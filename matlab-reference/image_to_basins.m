function [r, info] = image_to_basins(imageFile, varargin)
%IMAGE_TO_BASINS Construct and plot a rational map's attracting basins.
%   [R, INFO] = IMAGE_TO_BASINS('regions.png', 'NumBasins', 3, ...
%       'GridSize', 500, 'InnerIterations', 8, 'MaxIterations', 30)
%   returns a function handle R(Z) for scalars or arrays of complex points
%   (including Inf). The image is centered at zero with its longer side
%   of length 2. The top of the image corresponds to positive Im(z).
%
%   The construction is based on Section 2 of Fisher, Hill, Lazebnik and
%   Thompson, Simultaneous Approximation by Attracting Basins (2026).
%   https://arxiv.org/abs/2605.09903
%
%   A quadrature of Cauchy integrals gives a Runge approximant
%       S(z) = constant + sum(residues(k)/(z - poles(k))).
%   With d = NumBasins, F(u) = u^d / (1 + d/(d-1)*u^(d-1)) is the
%   paper's polynomial in reciprocal coordinates. The returned map is
%       r(z) = anchorOrigin + anchorScale * F^(n)(S(z)),
%   where F^(n) means n compositions. Small auxiliary regions to the
%   right of the image put all d attracting fixed points in trapping disks.
%   The evaluator keeps the map in composed form.
%
%   Image and approximation options (name-value pairs):
%     NumBasins        d >= 2. Default uses all distinct colors if there
%                      are at most 8, or clusters RGB values into 3 colors.
%     Palette          Optional d-by-3 RGB values in [0,1] or [0,255].
%                      Labels follow row order. Pixels use the nearest RGB.
%     OutsideLabel     Label extending outside the image, even if absent
%                      from it. Default is the most common border label.
%     BoundaryColors   Optional RGB rows to exclude as boundary pixels.
%     ColorTolerance   Euclidean RGB tolerance for BoundaryColors (1/255).
%     AlphaBackground  RGB background for transparent pixels ([1 1 1]).
%     ImageSize        Maximum input raster side in pixels (128).
%     BoundaryPixels   Margin for the interior fit diagnostic (2). Also
%                      sets the default Cauchy quadrature panel length.
%     PanelPixels      Panel length in input pixels (default 2*max(1,margin)).
%     QuadratureOrder  Gauss nodes per panel in automatic degree mode (4).
%     RationalDegree   Exact degree of S, or [] to use the panel rule.
%                      A fixed degree allocates nodes by edge length and
%                      needs at least one per merged edge, plus anchor poles.
%     AnchorDegree     Number of poles around each nonexterior anchor (16).
%     AnchorScale      Size of the auxiliary region cluster (0.10).
%     RungeShift       Small complex additive perturbation of S (0).
%     InnerIterations  n in F^(n). Default starts at 6 and increases if
%                      needed for the trapping disks. The degree of r is
%                      RationalDegree * d^n using the actual degree of S.
%
%   Rendering and diagnostics:
%     GridSize         Longest plot side (360), or [rows columns].
%     PlotBounds       [xmin xmax ymin ymax]. Default is the image extent.
%     MaxIterations    Maximum r iterates per plotted point (25).
%     CriticalCheckMaxDegree
%                      Numerically check critical-value orbits if degree(S)
%                      is at most this value (180). Use 0 to skip the check.
%     UnresolvedColor  RGB used when no trap was reached ([.30 .30 .30]).
%     ShowPlot         Show target regions and basins side by side (true).
%     OutputImage      Optional filename for saving the basin raster.
%     Verbose          Print the fit and orbit diagnostics (true).
%
%   INFO.runge evaluates S. INFO contains its poles, residues, and constant,
%   the trapping disks and fixed points, and the label and iteration grids.
%   Label 0 means unresolved. INFO.coreCaptureFraction measures one-step
%   capture in the intended trap at the sampled interior pixel centers.
%   The critical-value check is numerical. The paper's Hausdorff theorem
%   requires regions sharing a common boundary. Base MATLAB suffices.
%
%   Revision notes (changes from the first version):
%     * RationalDegree mode: node allocation breaks ties by edge order, so
%       the pole set no longer depends on rounding in linspace and sum.
%     * postBound returns Inf, not a negative number, when the disk around
%       the zero target reaches the pole of F.
%     * The critical check counts near-infinite eigenvalues with the multiple
%       critical point at infinity (INFO.criticalCheck.nearInfinity).
%   The Python package attracting_basins adds what this file does not have:
%   certified basin cells, a complete critical-point count at any degree,
%   and fixed-point multipliers.

    opts = parseOptions(imageFile, varargin{:});
    [targetLabels, palette, outside, xEdges, yEdges] = readRegions(imageFile, opts);
    labels = fillBoundaryPixels(targetLabels);
    d = size(palette, 1);

    % F has the fixed critical points 0 and the roots of u^(d-1) = -(d-1).
    otherLabels = find((1:d).' ~= outside);
    targets = zeros(d, 1);
    angles = (2*(0:d-2).' + 1)*pi/(d-1);
    targets(otherLabels) = (d-1)^(1/(d-1)) * exp(1i*angles);
    distances = abs(targets - targets.');
    distances(1:d+1:end) = Inf;
    spacing = min(distances(:));
    pixelSize = min(xEdges(2)-xEdges(1), yEdges(1)-yEdges(2));
    scale = opts.AnchorScale;
    origin = xEdges(end) + scale*(1+max(abs(targets))) + 2*pixelSize;
    centers = origin + scale*targets;
    anchorRadius = 0.22*scale*spacing;

    values = reshape(targets(labels(:)), size(labels));
    [starts, ends, jumps] = rasterEdges(values, xEdges, yEdges);
    panelPixels = opts.PanelPixels;
    if isempty(panelPixels)
        panelPixels = 2*max(1, opts.BoundaryPixels);
    end
    anchorCount = numel(otherLabels)*opts.AnchorDegree;
    budget = opts.RationalDegree;
    if ~isempty(budget)
        budget = budget - anchorCount;
    end
    [poles, residues] = edgeQuadrature(starts, ends, jumps, ...
        panelPixels*pixelSize, opts.QuadratureOrder, budget, anchorCount);

    % On an anchor circle the trapezoid rule gives 1/(1-((z-c)/h)^m).
    offsets = anchorRadius*exp(2i*pi*(0:opts.AnchorDegree-1).'/opts.AnchorDegree);
    anchorPoles = offsets + centers(otherLabels).';
    anchorResidues = -offsets*targets(otherLabels).'/opts.AnchorDegree;
    poles = [poles; anchorPoles(:)];
    residues = [residues; anchorResidues(:)];
    if numel(unique(poles)) ~= numel(poles)
        error('image_to_basins:PoleCollision', ...
            'Some poles coincide in floating point. Increase AnchorScale or lower RationalDegree.');
    end
    S = @(z) evaluateRunge(z, poles, residues, opts.RungeShift);
    [trapRadii, n] = trappingDisks(S, poles, residues, ...
        targets, centers, scale, spacing, opts.InnerIterations);
    r = @(z) evaluateMap(z, S, d, n, origin, scale);

    fixedPoints = centers;
    for k = 1:40
        next = r(fixedPoints);
        converged = all(abs(next-fixedPoints) <= 16*eps*(1+abs(next)));
        fixedPoints = next;
        if converged
            break
        end
    end

    imageBounds = [xEdges(1), xEdges(end), yEdges(end), yEdges(1)];
    bounds = opts.PlotBounds;
    if isempty(bounds)
        bounds = imageBounds;
    end
    [x, y] = plotGrid(bounds, opts.GridSize);
    [X, Y] = meshgrid(x, y);
    [basins, iterations] = classifyOrbits(complex(X, Y), r, centers, trapRadii, opts.MaxIterations);
    [coreError, coreCapture, coreCount] = checkImageCores( ...
        targetLabels, targets, xEdges, yEdges, opts.BoundaryPixels, ...
        S, d, n, scale, trapRadii);
    critical = checkCriticalValues(poles, residues, r, centers, trapRadii, opts);

    info = struct('runge', S, 'poles', poles, 'residues', residues, ...
        'constant', opts.RungeShift, 'rationalDegree', numel(poles), ...
        'polynomialDegree', d, 'mapDegree', numel(poles)*d^n, ...
        'innerIterations', n, 'anchorOrigin', origin, 'anchorScale', scale, ...
        'rungeTargets', targets, 'trapCenters', centers, 'trapRadii', trapRadii, ...
        'fixedPoints', fixedPoints, ...
        'fixedPointResiduals', abs(r(fixedPoints)-fixedPoints), ...
        'palette', palette, 'outsideLabel', outside, 'targetLabels', targetLabels, ...
        'imageBounds', imageBounds, 'plotBounds', bounds, 'x', x, 'y', y, ...
        'basinLabels', basins, 'entryIterations', iterations, ...
        'unresolvedFraction', nnz(basins == 0)/numel(basins), ...
        'coreMaxRungeError', coreError, 'coreCaptureFraction', coreCapture, ...
        'coreSampleCount', coreCount, 'criticalCheck', critical);

    if opts.Verbose
        fprintf('degree(S) = %d, degree(F) = %d, n = %d, degree(r) = %.5g\n', ...
            info.rationalDegree, d, n, info.mapDegree);
        fprintf('Grid unresolved: %.2f%%. ', 100*info.unresolvedFraction);
        if coreCount > 0
            fprintf('Target interior samples sent to the intended trap: %.2f%% (%d samples).\n', ...
                100*coreCapture, coreCount);
        else
            fprintf('No interior samples at the requested BoundaryPixels.\n');
        end
        fprintf('Critical values: %s\n', critical.status);
    end
    if opts.ShowPlot || ~isempty(opts.OutputImage)
        plotPalette = [asRGB(opts.UnresolvedColor); palette];
        basinRGB = colorRaster(basins, plotPalette);
        if ~isempty(opts.OutputImage)
            imwrite(basinRGB, opts.OutputImage);
        end
        if opts.ShowPlot
            figure('Name', 'Attracting basins', 'Color', 'w');
            imagePanel(subplot(1,2,1), colorRaster(targetLabels, plotPalette), ...
                imageBounds, 'Target regions');
            imagePanel(subplot(1,2,2), basinRGB, bounds, 'Attracting basins');
        end
    end
end

function opts = parseOptions(imageFile, varargin)
    positiveInt = @(v) isnumeric(v) && isreal(v) && isscalar(v) ...
        && isfinite(v) && v >= 1 && v == fix(v);
    optionalInt = @(v) isempty(v) || positiveInt(v);
    atLeastTwo = @(v) positiveInt(v) && v >= 2;
    nonnegativeInt = @(v) isnumeric(v) && positiveInt(double(v)+1);
    positiveReal = @(v) isnumeric(v) && isreal(v) && isscalar(v) ...
        && isfinite(v) && v > 0;
    rgbRows = @(v) isnumeric(v) && isreal(v) && ismatrix(v) && size(v,2) == 3 ...
        && all(isfinite(v(:)) & v(:) >= 0 & v(:) <= 255);
    rgbRow = @(v) rgbRows(v) && size(v,1) == 1;
    flag = @(v) (islogical(v) || isnumeric(v)) && isscalar(v) && (v == 0 || v == 1);
    filename = @(v) (ischar(v) && (isrow(v) || isempty(v))) || (isstring(v) && isscalar(v));
    assert(filename(imageFile) && ~isempty(char(imageFile)), ...
        'Provide an image filename as the first argument.');

    p = inputParser;
    p.FunctionName = 'image_to_basins';
    addParameter(p, 'NumBasins', [], @(v) isempty(v) || atLeastTwo(v));
    addParameter(p, 'Palette', [], @(v) isempty(v) || (rgbRows(v) && size(v,1) >= 2));
    addParameter(p, 'OutsideLabel', [], optionalInt);
    addParameter(p, 'BoundaryColors', [], @(v) isempty(v) || rgbRows(v));
    addParameter(p, 'ColorTolerance', 1/255, @(v) isnumeric(v) && isreal(v) ...
        && isscalar(v) && isfinite(v) && v >= 0);
    addParameter(p, 'AlphaBackground', [1 1 1], rgbRow);
    addParameter(p, 'ImageSize', 128, atLeastTwo);
    addParameter(p, 'BoundaryPixels', 2, nonnegativeInt);
    addParameter(p, 'PanelPixels', [], @(v) isempty(v) || positiveReal(v));
    addParameter(p, 'QuadratureOrder', 4, positiveInt);
    addParameter(p, 'RationalDegree', [], optionalInt);
    addParameter(p, 'AnchorDegree', 16, atLeastTwo);
    addParameter(p, 'AnchorScale', 0.10, positiveReal);
    addParameter(p, 'RungeShift', 0, @(v) isnumeric(v) && isscalar(v) && isfinite(v));
    addParameter(p, 'InnerIterations', [], optionalInt);
    addParameter(p, 'GridSize', 360, @(v) isnumeric(v) && isreal(v) && isvector(v) ...
        && numel(v) <= 2 && all(isfinite(v) & v >= 2 & v == fix(v)));
    addParameter(p, 'PlotBounds', [], @(v) isempty(v) || ...
        (isnumeric(v) && isreal(v) && isvector(v) && numel(v) == 4 ...
        && all(isfinite(v)) && v(1) < v(2) && v(3) < v(4)));
    addParameter(p, 'MaxIterations', 25, nonnegativeInt);
    addParameter(p, 'CriticalCheckMaxDegree', 180, nonnegativeInt);
    addParameter(p, 'UnresolvedColor', [.30 .30 .30], rgbRow);
    addParameter(p, 'ShowPlot', true, flag);
    addParameter(p, 'OutputImage', '', filename);
    addParameter(p, 'Verbose', true, flag);
    parse(p, varargin{:});
    opts = p.Results;
    for name = fieldnames(opts).'
        if isnumeric(opts.(name{1}))
            opts.(name{1}) = double(opts.(name{1}));
        end
    end
    opts.PlotBounds = opts.PlotBounds(:).';
    opts.OutputImage = char(opts.OutputImage);
end

function [labels, palette, outside, xEdges, yEdges] = readRegions(filename, opts)
    [raw, map, alpha] = imread(char(filename));
    [height, width, channels] = size(raw);
    extent = max(height, width);
    resized = max(1, round([height width]*min(1, opts.ImageSize/extent)));
    row = floor(((1:resized(1))-.5)*height/resized(1))+1;
    col = floor(((1:resized(2))-.5)*width/resized(2))+1;
    raw = raw(row, col, :);
    if ~isempty(map)
        index = double(raw) + ~isfloat(raw);
        rgb = map(index(:), :);
    elseif channels == 1 || channels == 3
        rgb = reshape(unitImage(raw), [], channels);
        if channels == 1
            rgb = repmat(rgb, 1, 3);
        end
    else
        error('image_to_basins:ImageFormat', 'Use an RGB, grayscale, or indexed image.');
    end
    if ~isempty(alpha)
        if isequal(size(alpha), [height width])
            opacity = alpha(row, col);
        elseif ~isempty(map) && numel(alpha) == size(map,1)
            opacity = alpha(index);
        else
            error('image_to_basins:AlphaSize', 'The image transparency has an unsupported size.');
        end
        opacity = unitImage(opacity(:));
        rgb = opacity.*rgb + (1-opacity).*asRGB(opts.AlphaBackground);
    end

    known = true(size(rgb,1),1);
    if ~isempty(opts.BoundaryColors)
        [~, gap] = nearestColor(rgb, asRGB(opts.BoundaryColors));
        known = gap > opts.ColorTolerance^2;
    end
    if ~any(known)
        error('image_to_basins:NoColors', 'Every pixel was excluded by BoundaryColors.');
    end
    palette = asRGB(opts.Palette);
    if isempty(palette)
        [colors, ~, index] = unique(rgb(known,:), 'rows');
        counts = accumarray(index, 1);
        d = opts.NumBasins;
        if isempty(d)
            d = size(colors,1);
            if d > 8
                d = 3;
            end
        end
        if d < 2 || d > size(colors,1)
            error('image_to_basins:ColorCount', ...
                'Choose at least two available colors, or supply a Palette with an exterior color.');
        end
        palette = clusterColors(colors, counts, d);
    elseif ~isempty(opts.NumBasins) && opts.NumBasins ~= size(palette,1)
        error('image_to_basins:PaletteSize', 'NumBasins must match the number of Palette rows.');
    end
    if size(unique(palette, 'rows'),1) ~= size(palette,1)
        error('image_to_basins:DuplicateColor', 'Palette rows must be distinct.');
    end

    labels = zeros(size(known));
    labels(known) = nearestColor(rgb(known,:), palette);
    labels = reshape(labels, resized);
    outside = opts.OutsideLabel;
    if isempty(outside)
        border = true(resized);
        border(2:end-1,2:end-1) = false;
        borderLabels = labels(border & labels > 0);
        if isempty(borderLabels)
            borderLabels = labels(labels > 0);
        end
        outside = mode(borderLabels);
    end
    if outside > size(palette,1)
        error('image_to_basins:OutsideLabel', 'OutsideLabel exceeds the number of colors.');
    end
    used = false(size(palette,1),1);
    used(labels(labels > 0)) = true;
    used(outside) = true;
    if any(~used)
        error('image_to_basins:UnusedColor', ...
            'Palette label(s) %s have no pixels. Check Palette, NumBasins, or ImageSize.', ...
            mat2str(find(~used).'));
    end
    xEdges = linspace(-width/extent, width/extent, resized(2)+1);
    yEdges = linspace(height/extent, -height/extent, resized(1)+1);
end

function rgb = asRGB(rgb)
    rgb = double(rgb);
    if any(rgb(:) > 1)
        rgb = rgb/255;
    end
end

function a = unitImage(a)
    if isinteger(a)
        a = double(a)/double(intmax(class(a)));
    else
        a = double(a);
    end
end

function [labels, best] = nearestColor(pixels, palette)
    best = inf(size(pixels,1),1);
    labels = zeros(size(best));
    for j = 1:size(palette,1)
        dist = sum((pixels-palette(j,:)).^2, 2);
        closer = dist < best;
        labels(closer) = j;
        best(closer) = dist(closer);
    end
end

function palette = clusterColors(colors, weights, d)
    if size(colors,1) == d
        palette = colors;
        return
    end
    palette = zeros(d,3);
    [~, seed] = max(weights);
    gap = inf(size(weights));
    for j = 1:d
        palette(j,:) = colors(seed,:);
        gap = min(gap, sum((colors-palette(j,:)).^2, 2));
        [~, seed] = max(weights.*gap);
    end
    previous = [];
    for k = 1:30
        groups = nearestColor(colors, palette);
        if isequal(groups, previous)
            break
        end
        previous = groups;
        for j = 1:d
            select = groups == j;
            if any(select)
                palette(j,:) = weights(select).'*colors(select,:)/sum(weights(select));
            end
        end
    end
end

function labels = fillBoundaryPixels(labels)
    % Extend the known labels by distance in the four-neighbor pixel grid.
    [rows, cols] = size(labels);
    while any(labels(:) == 0)
        up = [zeros(1,cols); labels(1:end-1,:)];
        down = [labels(2:end,:); zeros(1,cols)];
        left = [zeros(rows,1), labels(:,1:end-1)];
        right = [labels(:,2:end), zeros(rows,1)];
        neighbors = {up, down, left, right};
        for j = 1:numel(neighbors)
            fill = labels == 0 & neighbors{j} > 0;
            labels(fill) = neighbors{j}(fill);
        end
    end
end

function [starts, ends, jumps] = rasterEdges(values, x, y)
    % Oriented contour edges have the left value minus the right value.
    % This sums the Cauchy integrals of all labels, including holes.
    [rows, cols] = size(values);
    h = [zeros(1,cols); values] - [values; zeros(1,cols)];
    v = [zeros(rows,1), values] - [values, zeros(rows,1)];
    starts = cell(rows+cols+2,1);
    ends = starts;
    jumps = starts;
    for j = 1:rows+1
        [first, last, weight] = nonzeroRuns(h(j,:));
        starts{j} = x(first).' + 1i*y(j);
        ends{j} = x(last+1).' + 1i*y(j);
        jumps{j} = weight;
    end
    for j = 1:cols+1
        [first, last, weight] = nonzeroRuns(v(:,j).');
        k = rows+1+j;
        starts{k} = x(j) + 1i*y(last+1).';
        ends{k} = x(j) + 1i*y(first).';
        jumps{k} = weight;
    end
    starts = vertcat(starts{:});
    ends = vertcat(ends{:});
    jumps = vertcat(jumps{:});
end

function [first, last, weights] = nonzeroRuns(row)
    changes = [1, find(row(2:end) ~= row(1:end-1))+1, numel(row)+1];
    first = changes(1:end-1);
    last = changes(2:end)-1;
    weights = row(first).';
    keep = weights ~= 0;
    first = first(keep);
    last = last(keep);
    weights = weights(keep);
end

function [poles, residues] = edgeQuadrature(starts, ends, jumps, panelLength, order, budget, anchorCount)
    lengths = abs(ends-starts);
    if isempty(budget)
        panels = max(1, ceil(lengths/panelLength - 32*eps(lengths/panelLength)));
        counts = order*panels;
    else
        minimum = numel(lengths);
        if budget < minimum
            error('image_to_basins:DegreeBudget', ...
                'RationalDegree must be at least %d for this raster and AnchorDegree.', ...
                minimum+anchorCount);
        end
        % Equal-length edges have tied shares. Round before the (stable) sort
        % so that ties go by edge order, not by the last bits of linspace/sum.
        share = round((budget-minimum)*lengths/sum(lengths)*1e9)/1e9;
        counts = 1+floor(share);
        [~, priority] = sort(round((share-floor(share))*1e9), 'descend');
        extra = priority(1:(budget-sum(counts)));
        counts(extra) = counts(extra)+1;
        panels = max(1, floor(counts/order));
    end
    poles = complex(zeros(sum(counts),1));
    residues = poles;
    rules = cell(2, 2*order-1);
    next = 1;
    for j = 1:numel(lengths)
        base = floor(counts(j)/panels(j));
        remainder = mod(counts(j), panels(j));
        step = (ends(j)-starts(j))/panels(j);
        for k = 1:panels(j)
            q = base + (k <= remainder);
            if isempty(rules{1,q})
                [rules{1,q}, rules{2,q}] = gaussRule(q);
            end
            index = next:next+q-1;
            poles(index) = starts(j) + step*((k-1)+rules{1,q});
            residues(index) = -jumps(j)*step*rules{2,q}/(2*pi*1i);
            next = next+q;
        end
    end
end

function [nodes, weights] = gaussRule(n)
    if n == 1
        nodes = .5;
        weights = 1;
        return
    end
    k = 1:n-1;
    beta = k./sqrt(4*k.^2-1);
    [vectors, eigenvalues] = eig(diag(beta,1)+diag(beta,-1));
    [nodes, idx] = sort((diag(eigenvalues)+1)/2);
    weights = vectors(1,idx).'.^2;
end

function value = evaluateRunge(z, poles, residues, shift)
    shape = size(z);
    z = double(z(:));
    value = complex(nan(size(z)));
    value(isinf(z) & ~isnan(z)) = shift;
    finite = find(isfinite(z));
    % Bound the size of the temporary Cauchy matrix.
    chunkSize = max(1, min(8192, floor(750000/numel(poles))));
    for start = 1:chunkSize:numel(finite)
        rows = finite(start:min(start+chunkSize-1,numel(finite)));
        differences = z(rows)-poles.';
        v = shift + (1./differences)*residues;
        bad = find(~isfinite(v));
        exactPole = any(differences(bad,:) == 0, 2);
        v(bad(exactPole)) = Inf;
        value(rows) = v;
    end
    value = reshape(value, shape);
end

function value = evaluateMap(z, S, d, n, origin, scale)
    value = iteratePostMap(S(z), d, n);
    finite = isfinite(value);
    value(finite) = origin + scale.*value(finite);
end

function u = iteratePostMap(u, d, n)
    factor = d/(d-1);
    for k = 1:n
        finite = isfinite(u);
        small = finite & abs(u) <= 1;
        large = finite & ~small;
        v = u(small);
        den = 1 + factor*v.^(d-1);
        next = v.^d./den;
        next(den == 0) = Inf;
        u(small) = next;
        v = u(large);
        den = (1./v).^(d-1) + factor;
        next = v./den;
        next(den == 0) = Inf;
        u(large) = next;
    end
end

function [radii, n] = trappingDisks(S, poles, residues, targets, centers, scale, spacing, requested)
    d = numel(targets);
    baseRadius = min(.25, spacing/3)*ones(d,1);
    while any(postBound(targets, baseRadius, d) > baseRadius/2)
        baseRadius = baseRadius/2;
    end
    dist = abs(centers-poles.');
    maxRadius = min(0.3*min(dist,[],2), 0.22*scale*spacing);
    slopeBound = (1./(dist.*(dist-maxRadius)))*abs(residues);
    slack = 32*eps*(1+(1./dist)*abs(residues));
    centerError = abs(S(centers)-targets)+slack;
    if any(centerError >= baseRadius)
        error('image_to_basins:AnchorFit', ...
            'Runge fit misses an anchor disk. Increase RationalDegree, QuadratureOrder, or AnchorScale, or reduce RungeShift.');
    end
    radii = min(maxRadius, 0.85*(baseRadius-centerError)./slopeBound);
    if isempty(requested)
        n = 6;
    else
        n = requested;
    end
    bound = centerError + radii.*slopeBound;
    for k = 1:n
        bound = postBound(targets, bound, d);
    end
    if isempty(requested)
        while any(scale*bound > radii/2)
            n = n+1;
            bound = postBound(targets, bound, d);
        end
    else
        % A smaller disk may already trap for a short prescribed iterate.
        for attempt = 1:20
            needsShrink = scale*bound > radii/2;
            if ~any(needsShrink)
                break
            end
            radii(needsShrink) = radii(needsShrink)/2;
            bound = centerError + radii.*slopeBound;
            for k = 1:n
                bound = postBound(targets, bound, d);
            end
        end
        if any(scale*bound > radii/2)
            error('image_to_basins:InnerIterations', ...
                'This InnerIterations value did not establish trapping disks. Increase it or use [].');
        end
    end
end

function bound = postBound(targets, radius, d)
    % Bound |F(u)-target| on |u-target| <= radius. For a nonzero target
    % write u = target*(1+t) and use the double zero of F(u)-target.
    zero = targets == 0;
    bound = inf(size(targets));
    r0 = radius(zero);
    den0 = 1 - d/(d-1)*r0.^(d-1);
    b0 = r0.^d./den0;
    b0(den0 <= 0) = Inf;    % the pole of F lies inside the disk
    bound(zero) = b0;
    ratio = radius(~zero)./abs(targets(~zero));
    den = d-1 - d*expm1((d-1)*log1p(ratio));
    term = ones(size(ratio));
    numerator = zeros(size(ratio));
    for k = 1:d
        term = term.*ratio*((d-k+1)/k);
        numerator = numerator+(k-1)*term;
    end
    b = abs(targets(~zero)).*numerator./den;
    b(den <= 0) = Inf;
    bound(~zero) = b;
end

function [x, y] = plotGrid(bounds, requested)
    spans = [bounds(4)-bounds(3), bounds(2)-bounds(1)];
    if isscalar(requested)
        requested = max(2, round(requested*spans/max(spans)));
    end
    x = bounds(1)+((1:requested(2))-.5)*spans(2)/requested(2);
    y = bounds(4)-((1:requested(1))-.5)*spans(1)/requested(1);
end

function [labels, entry] = classifyOrbits(z, r, centers, radii, maxIterations)
    labels = zeros(size(z));
    entry = nan(size(z));
    active = (1:numel(z)).';
    values = z(:);
    for k = 0:maxIterations
        hits = zeros(size(values));
        for j = 1:numel(centers)
            hits(abs(values-centers(j)) <= radii(j)) = j;
        end
        trapped = hits > 0;
        labels(active(trapped)) = hits(trapped);
        entry(active(trapped)) = k;
        keep = ~trapped & ~isnan(values);
        if k == maxIterations || ~any(keep)
            break
        end
        active = active(keep);
        values = r(values(keep));
    end
end

function [maxError, captured, count] = checkImageCores(labels, targets, x, y, margin, S, d, n, scale, radii)
    core = false(size(labels));
    if margin == 0
        core = labels > 0;
    elseif 2*margin+1 <= min(size(labels))
        kernel = ones(2*margin+1, 1);
        for j = 1:d
            count = conv2(kernel, kernel.', double(labels == j), 'same');
            core = core | (count == numel(kernel)^2);
        end
    end
    [row, col] = find(core);
    count = numel(row);
    maxError = NaN;
    captured = NaN;
    if count == 0
        return
    end
    z = complex((x(col)+x(col+1))/2, (y(row)+y(row+1))/2);
    u = S(z(:));
    ideal = targets(labels(core));
    maxError = max(abs(u-ideal));
    inTrap = scale*abs(iteratePostMap(u,d,n)-ideal) <= radii(labels(core));
    captured = nnz(inTrap)/count;
end

function check = checkCriticalValues(poles, residues, r, centers, radii, opts)
    m = numel(poles);
    check = struct('performed', false, 'status', 'skipped', 'rungeCriticalPoints', [], ...
        'relativeDerivativeResidual', [], 'valuesTested', [], ...
        'basinLabels', [], 'allComputedCaptured', [], 'nearInfinity', []);
    if m > opts.CriticalCheckMaxDegree
        check.status = sprintf('skipped (degree %d exceeds check limit %d).', ...
            m, opts.CriticalCheckMaxDegree);
        return
    end
    % The finite eigenvalues are zeros of sum residues./(z-poles).^2.
    % The arrowhead pencil uses a double-pole Jordan block for each pole.
    D = diag(poles);
    A = [D, eye(m), zeros(m,1); zeros(m), D, ones(m,1); ...
        residues.'/max(abs(residues)), zeros(1,m+1)];
    B = diag([ones(2*m,1); 0]);
    try
        eigenvalues = eig(A,B);
    catch err
        check.status = sprintf('eigenvalue calculation failed (%s).', err.message);
        return
    end
    points = eigenvalues(isfinite(eigenvalues));
    % Infinity is a multiple critical point of S by construction: Gauss panels
    % with q nodes and closed contours give sum(residues.*poles.^k) = 0 for
    % k < 2q. QZ returns part of it as huge finite eigenvalues. They are
    % counted with infinity, whose critical value r(Inf) is tested below.
    far = abs(points) > 1e3*max(abs(poles));
    nearInfinity = 2*m-2-nnz(~far);
    points = points(~far);
    terms = residues.'./(points-poles.').^2;
    residual = abs(sum(terms,2))./sum(abs(terms),2);
    % Infinity is always tested: it is a critical point of S (see above).
    values = [r(points); r(Inf); centers];
    labels = classifyOrbits(values, r, centers, radii, opts.MaxIterations);
    check.performed = true;
    check.rungeCriticalPoints = points;
    check.relativeDerivativeResidual = residual;
    check.valuesTested = values;
    check.basinLabels = labels;
    check.allComputedCaptured = all(labels > 0);
    check.nearInfinity = nearInfinity;
    check.status = sprintf(['%d/%d tested values captured. %d/%d finite roots have residual <= 1e-6. ' ...
        '%d of %d critical points at or near infinity (value S(Inf) tested).'], ...
        nnz(labels), numel(labels), nnz(residual <= 1e-6), numel(residual), nearInfinity, 2*m-2);
end

function rgb = colorRaster(labels, palette)
    rgb = reshape(palette(labels(:)+1,:), [size(labels), 3]);
end

function imagePanel(ax, rgb, bounds, caption)
    dx = (bounds(2)-bounds(1))/size(rgb,2);
    dy = (bounds(4)-bounds(3))/size(rgb,1);
    image(ax, bounds(1:2)+[.5 -.5]*dx, bounds(3:4)+[.5 -.5]*dy, flipud(rgb));
    set(ax, 'YDir', 'normal');
    axis(ax, 'image');
    xlim(ax, bounds(1:2));
    ylim(ax, bounds(3:4));
    xlabel(ax, 'Re(z)');
    ylabel(ax, 'Im(z)');
    title(ax, caption);
end
