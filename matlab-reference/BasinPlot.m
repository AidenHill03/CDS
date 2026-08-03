function [M, x, y] = BasinPlot(func, attractors, varargin)
%BASINPLOT  Classify each pixel by which attractor (cycle) it converges to.
%
%   M = BasinPlot(func, attractors)
%   M = BasinPlot(func, attractors, 'Name', Value, ...)
%
%   func         @(z) the iterated map -- typically state.func from
%                CompileFamily(family)(a).
%   attractors   EITHER a numeric vector of attracting points (the legacy
%                interface; each entry is a period-1 "cycle", Inf allowed)
%                OR a cell array of numeric vectors, one per attracting
%                cycle, as returned by FindAttractors (Inf entries
%                allowed). A pixel is assigned to cycle k the first time
%                its orbit comes chordally within Tol of ANY point of
%                cycle k.
%
%   CLASSIFICATION IS CHORDAL (sphere-first). All convergence tests use
%   the chordal metric on the Riemann sphere (see chordalDist), under
%   which Inf is an ordinary point. Consequences:
%     * There is NO EscapeR parameter anymore. "Escaped" is not a
%       separate concept: chordalDist(z, Inf) < Tol IS the escape test
%       (equivalent to |z| > ~2/Tol), and it only applies if Inf is
%       actually among the supplied attractors -- exactly the sphere
%       viewpoint of Fisher-Hill-Lazebnik-Thompson, where the Fatou set
%       decomposes into attracting basins and Inf's basin is just one of
%       them, not a failure mode.
%     * Pixels that never approach any supplied attractor stay 0
%       ("unresolved"). For maps of the type constructed in that paper
%       (Fatou set = disjoint union of attracting basins), the
%       unresolved fraction should tend to 0 as MaxIter grows -- making
%       it a cheap diagnostic that a candidate map has the right global
%       structure. Persistent unresolved regions signal rotation domains
%       (Siegel/Herman), parabolic points, or missed attractors.
%
%   Name-Value options
%   -------------------
%     Center       complex center of the view window         (default 0)
%     Scale        half-width of the view window               (default 1.5)
%     Resolution   pixels per side                              (default 500)
%     MaxIter      max iterations before giving up (unresolved) (default 200)
%     Tol          CHORDAL distance counted as "converged"      (default 1e-6)
%     Colormap     colormap FUNCTION (or name) used to assign
%                  one distinct color per attractor              (default @lines)
%     Legend       colorbar with one tick per attractor          (default true)
%     Axes         existing axes handle to draw into              (default [])
%     Filename     export path (exportgraphics), if non-empty      (default '')
%     Visible      whether a newly-created figure is visible       (default true)
%
%   Returns
%   -------
%     M    resolution x resolution matrix of basin indices:
%            0 = unresolved, k = converged to attractors{k} (or
%            attractors(k) for the legacy vector form).
%     x,y  axis vectors.
%
%   Examples
%   --------
%     % Legacy form (fixed points + Inf), unchanged:
%     family = CompileFamily(@(z,a) z.^3 + a./z.^3);
%     state  = family(0.5);
%     BasinPlot(state.func, state.attractors);
%
%     % Cycle-aware form:
%     cycles = FindAttractors(state);
%     BasinPlot(state.func, cycles);
%
%   Performance note: iterates the WHOLE grid every step (no shrinking
%   logical mask) for the same reason as JuliaPlot/ParameterPlot --
%   full-array vectorized arithmetic beats per-iteration logical-indexed
%   assignment in MATLAB. Indexing is only used to write labels for
%   newly-resolved pixels.

    p = inputParser;
    addRequired(p, 'func',       @(f) isa(f,'function_handle'));
    addRequired(p, 'attractors', @(v) (isnumeric(v) || iscell(v)) && ~isempty(v));
    addParameter(p, 'Center',     0,      @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Scale',      1.5,    @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'Resolution', 500,    @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'MaxIter',    200,    @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'Tol',        1e-6,   @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'PreMap',     [],     @(v) isempty(v) || isa(v,'function_handle'));
    addParameter(p, 'Colormap',   @hsv,   @(v) isa(v,'function_handle') || ischar(v) || isstring(v));
    addParameter(p, 'Legend',     true,   @(v) islogical(v) || isnumeric(v));
    addParameter(p, 'Axes',       [],     @(v) isempty(v) || isgraphics(v,'axes'));
    addParameter(p, 'Filename',   '',     @(v) ischar(v) || isstring(v));
    addParameter(p, 'Visible',    true,   @(v) islogical(v) || isnumeric(v));
    parse(p, func, attractors, varargin{:});
    opt = p.Results;

    % ---- normalize attractors to a cell array of cycles ---------------------
    if isnumeric(attractors)
        av = attractors(:);
        cyc = cell(numel(av), 1);
        for k = 1:numel(av)
            cyc{k} = av(k);
        end
    else
        cyc = attractors(:);
    end
    nAttr = numel(cyc);

    % ---- build grid -----------------------------------------------------------
    x = real(opt.Center) + linspace(-opt.Scale, opt.Scale, opt.Resolution);
    y = imag(opt.Center) + linspace(-opt.Scale, opt.Scale, opt.Resolution);
    [X, Y] = meshgrid(x, y);
    Z = X + 1i*Y;

    % Optional preprocessing map R, applied ONCE before iteration. This
    % realizes a composition S = func^n o PreMap: the grid seeds the
    % dynamics through R first, then func is iterated. Used by the
    % image->AAA->basin pipeline, where PreMap is the AAA approximant r
    % and func is the model map Qhat (so S_n = Qhat^n o r). With no
    % PreMap, classification is of func's own basins from the raw grid.
    if ~isempty(opt.PreMap)
        Z = opt.PreMap(Z);
    end

    M        = zeros(size(Z));
    resolved = false(size(Z));

    % ---- iterate --------------------------------------------------------------
    for n = 1:opt.MaxIter
        Z = opt.func(Z);

        for k = 1:nAttr
            pts = cyc{k};
            near = false(size(Z));
            for j = 1:numel(pts)
                near = near | (chordalDist(Z, pts(j)) < opt.Tol);
            end
            newlyResolved = ~resolved & near;
            if any(newlyResolved(:))
                M(newlyResolved) = k;
                resolved = resolved | newlyResolved;
            end
        end

        if all(resolved(:))
            break
        end
    end

    % ---- render ---------------------------------------------------------------
    if isempty(opt.Axes)
        fig = figure('Visible', logical(opt.Visible));
        ax = axes(fig);
    else
        ax = opt.Axes;
        fig = ancestor(ax, 'figure');
    end

    imagesc(ax, x, y, M);
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal', 'tight', 'off');

    if ischar(opt.Colormap) || isstring(opt.Colormap)
        cmapFcn = str2func(char(opt.Colormap));
    else
        cmapFcn = opt.Colormap;
    end
    attrColors = cmapFcn(max(nAttr, 1));
    cmap = [0 0 0; attrColors];   % index 0 (unresolved) = black
    colormap(ax, cmap);
    caxis(ax, [-0.5, nAttr + 0.5]); %#ok<CAXIS>

    if opt.Legend
        cb = colorbar(ax);
        cb.Ticks = 0:nAttr;
        labels = cell(1, nAttr + 1);
        labels{1} = 'unresolved';
        for k = 1:nAttr
            labels{k+1} = cycleLabel(cyc{k});
        end
        cb.TickLabels = labels;
    end

    if strlength(string(opt.Filename)) > 0
        exportgraphics(fig, opt.Filename, 'Resolution', 300);
    end
end

% =========================================================================
function s = cycleLabel(pts)
    if numel(pts) == 1
        if isinf(pts)
            s = 'Inf';
        else
            s = sprintf('%.4g%+.4gi', real(pts), imag(pts));
        end
    else
        if any(isinf(pts))
            s = sprintf('%d-cycle (thru Inf)', numel(pts));
        else
            s = sprintf('%d-cycle @ %.3g%+.3gi', numel(pts), real(pts(1)), imag(pts(1)));
        end
    end
end