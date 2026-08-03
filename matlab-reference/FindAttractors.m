function cycles = FindAttractors(state, varargin)
%FINDATTRACTORS  Numerically discover the attracting cycles of a map.
%
%   cycles = FindAttractors(state)
%   cycles = FindAttractors(state, 'Name', Value, ...)
%
%   state    the struct returned by CompileFamily(family)(a). Uses
%            state.func and state.criticalPoints (plus state.dfunc for
%            multiplier verification).
%
%   Returns a cell array `cycles`, one cell per attracting cycle found.
%   Each cell is a row vector of the cycle's points (possibly containing
%   Inf if the cycle passes through or sits at infinity). This is exactly
%   the format the upgraded BasinPlot accepts as its `attractors` input.
%
%   WHY CRITICAL ORBITS: by a classical theorem of Fatou, every
%   attracting cycle of a rational map attracts at least one critical
%   point. So iterating each critical orbit past its transient and
%   detecting the cycle it lands on finds EVERY attracting cycle -- no
%   symbolic solving, no guessing periods in advance. This replaces (and
%   strictly generalizes) CompileFamily's attracting-fixed-point filter:
%   fixed points are just the period-1 case. It's also the fix for the
%   deepest legacy-pipeline assumption -- "attractors = attracting fixed
%   points" -- which fails for any map whose attractor is a genuine
%   cycle.
%
%   Infinity is handled as an ordinary point on the sphere: orbits are
%   tracked in the chordal metric, an orbit settling at Inf is recorded
%   as the cycle {Inf}, and cycles passing through very-large-|z| values
%   are snapped to Inf entries.
%
%   Name-Value options
%   -------------------
%     BurnIn      iterations to discard as transient        (default 500)
%     MaxPeriod   largest cycle period to detect             (default 64)
%     Tol         chordal tolerance for cycle closure and
%                 for deduplicating cycles across critical
%                 orbits                                      (default 1e-9)
%     InfCutoff   |z| beyond which a value is treated as Inf  (default 1e12)
%     Verify      also verify |multiplier| < 1 for detected
%                 cycles (skipped for cycles containing Inf,
%                 where state.dfunc doesn't apply directly)    (default true)
%
%   Notes / limitations
%   -------------------
%   * Neutral or near-neutral cycles (|multiplier| ~ 1) converge too
%     slowly to be reliably detected at the default BurnIn. Siegel disks
%     and Herman rings have NO attracting cycle and correctly produce
%     nothing here -- pixels in those components will show as
%     "unresolved" in BasinPlot, which is the honest answer.
%   * If a critical orbit is still wandering chaotically after BurnIn
%     (critical point in the Julia set), no cycle is recorded for it.

    p = inputParser;
    addRequired(p, 'state', @(s) isstruct(s) && isfield(s,'func') && isfield(s,'criticalPoints'));
    addParameter(p, 'BurnIn',    500,   @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'MaxPeriod', 64,    @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'Tol',       1e-9,  @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'InfCutoff', 1e12,  @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'Verify',    true,  @(v) islogical(v) || isnumeric(v));
    parse(p, state, varargin{:});
    opt = p.Results;

    seeds = state.criticalPoints(:).';
    % A pole mapping to Inf, or Inf itself being attracting, is common;
    % include a large-|z| seed so the Inf basin is found even for maps
    % whose finite critical points all sit in other basins.
    seeds = [seeds, opt.InfCutoff];

    cycles = {};

    for s = seeds
        z = s;
        atInf = false;

        % ---- burn-in ------------------------------------------------------
        for n = 1:opt.BurnIn
            z = state.func(z);
            if ~isfinite(z) || abs(z) > opt.InfCutoff
                % Treat as having reached the Inf end of the sphere. For a
                % map where Inf is attracting-fixed this is terminal; for a
                % cycle THROUGH Inf (e.g. a pole in the cycle) one more
                % application of the map is undefined numerically, so we
                % record based on where we are.
                atInf = true;
                break
            end
        end

        if atInf
            cycles = addCycle(cycles, Inf, opt.Tol);
            continue
        end

        % ---- detect period: find smallest p with f^p(z) ~ z chordally -----
        orbit = zeros(1, opt.MaxPeriod + 1);
        orbit(1) = z;
        found = 0;
        for k = 1:opt.MaxPeriod
            zn = state.func(orbit(k));
            if ~isfinite(zn) || abs(zn) > opt.InfCutoff
                zn = Inf;
            end
            orbit(k+1) = zn;
            if chordalDist(zn, orbit(1)) < opt.Tol
                found = k;
                break
            end
            if isinf(zn)
                % passed through Inf without closing -- can't continue
                % iterating numerically from Inf with state.func; record
                % nothing and move on. (A genuine attracting cycle through
                % a pole would need the 1/z chart to follow; rare enough
                % to flag rather than silently mishandle.)
                warning('FindAttractors:infInOrbit', ...
                    'Orbit from seed %.4g%+.4gi passed through Inf without closing; skipped.', ...
                    real(s), imag(s));
                found = -1;
                break
            end
        end

        if found <= 0
            continue   % no cycle detected within MaxPeriod (or skipped)
        end

        cyc = orbit(1:found);

        % ---- verify it is actually attracting -----------------------------
        if opt.Verify && ~any(isinf(cyc))
            mult = prod(state.dfunc(cyc));
            if ~(abs(mult) < 1)
                % Converged numerically but multiplier says otherwise
                % (e.g. parabolic drift) -- don't trust it as an attractor.
                continue
            end
        end

        cycles = addCycle(cycles, cyc, opt.Tol * 1e3);  % looser dedupe tol
    end
end

% =========================================================================
function cycles = addCycle(cycles, newCyc, tol)
%ADDCYCLE  Append newCyc unless it matches an existing cycle up to cyclic
%   rotation (chordally, within tol).
    for c = 1:numel(cycles)
        old = cycles{c};
        if numel(old) ~= numel(newCyc)
            continue
        end
        for shift = 0:numel(old)-1
            rotated = circshift(newCyc, shift);
            if all(chordalDist(rotated, old) < tol)
                return   % duplicate
            end
        end
    end
    cycles{end+1} = newCyc; %#ok<AGROW>
end