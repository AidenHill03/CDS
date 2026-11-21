function dump(img, out, varargin)
  [r, info] = image_to_basins(img, 'ShowPlot', false, varargin{:});
  rand('seed', 1); 
  zt = [linspace(-1.2,1.4,57) + 1i*linspace(0.9,-0.8,57), Inf, NaN, info.poles(1), info.trapCenters.'].';
  rz = r(zt); sz = info.runge(zt);
  poles = info.poles; residues = info.residues; trapRadii = info.trapRadii; n = info.innerIterations;
  fixedPoints = info.fixedPoints; basinLabels = info.basinLabels; entry = info.entryIterations;
  targetLabels = info.targetLabels; palette = info.palette; outside = info.outsideLabel;
  coreErr = info.coreMaxRungeError; coreCap = info.coreCaptureFraction; coreCount = info.coreSampleCount;
  crit = info.criticalCheck; critPerformed = crit.performed;
  if crit.performed
    critLabels = crit.basinLabels; critPoints = crit.rungeCriticalPoints; critValues = crit.valuesTested; critInf = crit.nearInfinity;
  else
    critLabels = []; critPoints = []; critValues = []; critInf = [];
  end
  save('-v7', out, 'zt','rz','sz','poles','residues','trapRadii','n','fixedPoints','basinLabels','entry', ...
       'targetLabels','palette','outside','coreErr','coreCap','coreCount','critPerformed','critLabels','critPoints','critValues','critInf');
end
