function d = chordalDist(z, w)
%CHORDALDIST  Chordal (spherical) distance on the Riemann sphere.
%
%   d = chordalDist(z, w)
%
%   Computes  d(z,w) = 2|z-w| / sqrt((1+|z|^2)(1+|w|^2)),  the chordal
%   metric on the sphere, elementwise over arrays (with scalar expansion).
%   Either argument may be Inf, in which case the correct limiting value
%   d(z,Inf) = 2 / sqrt(1+|z|^2) is used.
%
%   Why this metric: it makes Inf an ordinary point. In particular the
%   old two-test classification ("within Tol of a finite attractor" OR
%   "|z| > EscapeR") collapses into ONE test: chordal distance to the
%   attractor < tol. For the Inf attractor, chordalDist(z,Inf) < tol is
%   equivalent to |z| > sqrt(4/tol^2 - 1) ~ 2/tol, i.e. the escape-radius
%   condition falls out automatically.
%
%   Range: 0 <= d <= 2 (antipodal points are at distance 2).

    % broadcast scalars to a common size
    if isscalar(z) && ~isscalar(w)
        z = z * ones(size(w));
    elseif isscalar(w) && ~isscalar(z)
        w = w * ones(size(z));
    end

    d = zeros(size(z));

    zInf = isinf(z);
    wInf = isinf(w);

    both = zInf & wInf;
    d(both) = 0;

    zi = zInf & ~wInf;   % z = Inf, w finite
    d(zi) = 2 ./ sqrt(1 + abs(w(zi)).^2);

    wi = wInf & ~zInf;   % w = Inf, z finite
    d(wi) = 2 ./ sqrt(1 + abs(z(wi)).^2);

    fin = ~zInf & ~wInf;
    d(fin) = 2 * abs(z(fin) - w(fin)) ./ ...
        sqrt((1 + abs(z(fin)).^2) .* (1 + abs(w(fin)).^2));

    % Guard against NaN from Inf*0-style edge cases in the arithmetic
    % (e.g. overflow of |z|^2): a NaN here always means "very far from
    % any finite point", so clamp to the metric's diameter.
    d(isnan(d) & ~(isnan(z) | isnan(w))) = 2;
end