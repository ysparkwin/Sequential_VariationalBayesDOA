% Version 1.0: (01/21/2023)
% written by Yongsung Park

% Yongsung Park, Florian Meyer, & Peter Gerstoft
% MPL/SIO/UCSD
% yongsungpark@ucsd.edu / flmeyer@ucsd.edu / gerstoft@ucsd.edu
% noiselab.ucsd.edu

% Citation
% Y. Park, F. Meyer, and P. Gerstoft, "Graph-based sequential beamforming," J. Acoust. Soc. Am. 153(1), (2023).
% https://doi.org/10.1121/10.0016876
% F. Meyer, Y. Park, and P. Gerstoft, "Variational Bayesian estimation of time-varying DOAs," in Proc. IEEE FUSION (2020), pp. 1–6.
% https://doi.org/10.23919/FUSION45008.2020.9190217
% Y. Park, F. Meyer, and P. Gerstoft, "Learning-Aided Initialization for Variational Bayesian DOA Estimation," in Proc. IEEE ICASSP (2022), pp. 4938–4942.
% https://doi.org/10.1109/ICASSP43922.2022.9746180

%%
clear; clc;
close all;

dbstop if error;

% addpath([cd,'/_common'])

SNRlist = [20];
for nSNR = 1:length(SNRlist)
Nsim = 1;
for nsim=1:Nsim
Nrng=27; rng(Nrng+nsim)
% rng(nsim)
disp(['SNR',num2str(SNRlist(nSNR)),'_',num2str(nsim)])
% Environment parameters
c = 1500;       % speed of sound
f = 200;        % frequency
lambda = c/f;   % wavelength

% ULA-horizontal array configuration
Nsensor = 15;               % number of sensors
d = 1/2*lambda;             % intersensor spacing
q = (0:1:(Nsensor-1))';     % sensor numbering
xq = (q-(Nsensor-1)/2)*d;   % sensor locations

% sensor configuration structure
Sensor_s.Nsensor = Nsensor;
Sensor_s.lambda = lambda;
Sensor_s.d = d;
Sensor_s.q = q;
Sensor_s.xq = xq;

% signal generation parameters
SNR = SNRlist(nSNR);

% total number of snapshots
Nsnapshot = 50;

% range of angle space
thetalim = [-90 90];

theta_separation = 0.5;

% Angular search grid
theta = (thetalim(1):theta_separation:thetalim(2))';
Ntheta = length(theta);

% Design/steering matrix (Sensing matrix)
sin_theta = sind(theta);
sensingMatrix = exp(-1i*2*pi/lambda*xq*sin_theta.')/sqrt(Nsensor);

% Generate received signal
anglesTrue = [-70; -55; -40; 35; 50; 65]; % DOA of sources at first snapshot [deg]
anglesTracks = repmat(anglesTrue,[1,Nsnapshot]);
anglesTracks(3,:) = anglesTracks(3,1) - 2*anglesTracks(3,1)./(1+exp(-.1*(-Nsnapshot/2:-Nsnapshot/2+Nsnapshot-1)));
anglesTracks(4,:) = anglesTracks(4,1) - 1.00*(0:Nsnapshot-1)';
sinAnglesTracks = sind(anglesTracks); 
Nsources = numel(anglesTrue);

receivedSignal = zeros(Nsensor,Nsnapshot);
source_amp = zeros(Nsources,Nsnapshot);
for snapshot = 1:Nsnapshot
    % Source generation
    % Complex Gaussian with zero-mean
    source_amp(:,snapshot) = complex(randn(size(anglesTrue)),randn(size(anglesTrue)))/sqrt(2);
    Xsource = source_amp(:,snapshot);
    
    % Represenation matrix (steering matrix)
    transmitMatrix = exp( -1i*2*pi/lambda*xq*sinAnglesTracks(:,snapshot).' )/sqrt(Nsensor);
    
    % Received signal without noise
    receivedSignal(:,snapshot) = sum(transmitMatrix*diag(Xsource),2);
    
    % add noise to the signals
    rnl = 10^(-SNR/20)*norm(Xsource);
    nwhite = complex(randn(Nsensor,1),randn(Nsensor,1))/sqrt(2*Nsensor);
    e = nwhite * rnl;	% error vector
    receivedSignal(:,snapshot) = receivedSignal(:,snapshot) + e;
end
ActSrcInd = cell(1,50); ActSrcInd(:) = {(1:6).'};

for snapshot = 1:Nsnapshot
snapshot
%%  Original VALSE
    disp('Original VALSE is running ...')
    outputValse = VALSE( receivedSignal(:,snapshot), q, 1, receivedSignal(:,snapshot) );
%     outputValse.ospa = getOSPA(anglesTracks(:,snapshot),asind(-outputValse.freqs * lambda/( 2 * pi * d)),8,2);
%     outputValse.card = numel(outputValse.freqs);
    
    if exist('outputsValse','var')==0, outputsValse = []; end
    outputsValse = [outputsValse;outputValse];

%% Sequential VALSE
% 1/sqrt(148) = 0.0822 rad (sigma_r)
% asind( (1/sqrt(148)) / ((2*pi*198 * 1500/198/2)/1500) ) = 1.5 deg
% 1/power( (sind(1.5)* (2*pi*198 * 1500/198/2)/1500 ),2 ) = 148
    kappaAdd = 148;
    disp('SVALSE w/ new initialization is running ...')
    if snapshot==1, prior2 = []; end
    outputSValseNI = newSVALSEv1p00( receivedSignal(:,snapshot), q, 1, receivedSignal(:,snapshot), prior2 );
%     outputSValseNI.ospa = getOSPA(anglesTracks(:,snapshot),asind(-outputSValseNI.freqs * lambda/( 2 * pi * d)),8,2);
%     outputSValseNI.card = numel(outputSValseNI.freqs);
    
    rhoPriorExisting = 0.75;
    rhoPriorNonExisting = 0.10;
    prior2.mus = outputSValseNI.mus;
    prior2.kappas = 1./(1./outputSValseNI.kappas + 1/kappaAdd);      % new angle is old angle plus noise
    numPriorComponents = size(outputSValseNI.freqs,1);
    prior2.rho = rhoPriorNonExisting * ones(Nsensor,1);
    prior2.rho(1:numPriorComponents) = rhoPriorExisting;
    
    if exist('outputsSValseNI','var')==0, outputsSValseNI = []; end
    outputsSValseNI = [outputsSValseNI;outputSValseNI];

end
end
% vars = who();
% varnames = vars(contains(vars, 'outputs'));
% save(['D_CG6s_SNR',num2str(SNR),'_',num2str(nsim)],...
%     'anglesTracks',varnames{:});
end
% save(['results_CG6_',num2str(SNR)])
% save(['data_results'])

%% Plot CBF & VALSE
if exist('outputPlot','var') == 0, outputPlot = outputsValse; end
if exist('FoS','var') == 0, FoS = 18; end

figure(2);
set(gcf,'position',[530,100,560,420]);
imagesc(1:Nsnapshot,theta,-inf);
caxis([-20 0])

% load hotAndCold, colormap(cmap)
colormap parula

rtIndex = []; rtTheta = []; rtMu = [];
for index=1:Nsnapshot
    rTheta = asind(-outputPlot(index).freqs * lambda/( 2 * pi * d));
    rMu = abs(outputPlot(index).amps);
    rMu = 20*log10( rMu / max(rMu) );
    
    rtIndex = [rtIndex;index*ones(size(rMu))];
    rtTheta = [rtTheta;rTheta];
    rtMu = [rtMu;rMu];
end

% hold on; scatter(rtIndex,rtTheta,8,rtMu,'o','linewidth',1); hold off;
hold on; scatter(rtIndex,rtTheta,50,rtMu,'filled','o','linewidth',.5,...
    'MarkerEdgeColor','k'); hold off;

title('Non-sequential VALSE')
xlabel('Time step','interpreter','latex')
ylabel('DOA~[$^\circ$]','interpreter','latex')
box on
set(gca,'fontsize',FoS,'YDir','normal','TickLabelInterpreter','latex','YTick',-80:40:80)
axis([.5 Nsnapshot+.5 -90 90])
% set(gca,'YTickLabel','')

outputBeamformer = sensingMatrix' * receivedSignal;
figure(3);
set(gcf,'position',[1,100,560,420]);
imagesc(1:Nsnapshot,theta,10*log10( (abs(outputBeamformer).^2) ./ max((abs(outputBeamformer).^2),[],1)));
% load hotAndCold, colormap(cmap)
colormap parula
caxis([-20 0])
title('Conventional beamforming')
xlabel('Time step','interpreter','latex')
ylabel('DOA~[$^\circ$]','interpreter','latex')
box on
set(gca,'fontsize',FoS,'YDir','normal','TickLabelInterpreter','latex','YTick',-80:40:80)
axis([.5 Nsnapshot+.5 -90 90])

hold on;
for snapshot=1:50
    srcind = ActSrcInd{snapshot};
    plot(snapshot,anglesTracks(srcind,snapshot),'kx',...
        'linewidth',1.5,'markersize',8)
end
hold off

%% Plot SVALSE
outputPlot = outputsSValseNI;
Nsnapshot = numel(outputPlot);
if exist('FoS','var') == 0, FoS = 18; end

figure(1);
set(gcf,'position',[1060,100,560,420]);
imagesc(1:Nsnapshot,theta,-inf);
caxis([-20 0])

% load hotAndCold, colormap(cmap)
colormap parula

rtIndex = []; rtTheta = []; rtMu = [];
for index=1:Nsnapshot
    rTheta = asind(-outputPlot(index).freqs * lambda/( 2 * pi * d));
    rMu = abs(outputPlot(index).amps);
    rMu = 20*log10( rMu / max(rMu) );
    
    rtIndex = [rtIndex;index*ones(size(rMu))];
    rtTheta = [rtTheta;rTheta];
    rtMu = [rtMu;rMu];
end

% hold on; scatter(rtIndex,rtTheta,8,rtMu,'o','linewidth',1); hold off;
hold on; scatter(rtIndex,rtTheta,50,rtMu,'filled','o','linewidth',.5,...
    'MarkerEdgeColor','k'); hold off;

title('Sequential VALSE')
xlabel('Time step','interpreter','latex')
ylabel('DOA~[$^\circ$]','interpreter','latex')
box on
set(gca,'fontsize',FoS,'YDir','normal','TickLabelInterpreter','latex','YTick',-80:40:80)
axis([.5 Nsnapshot+.5 -90 90])
% set(gca,'YTickLabel','')

%%
% rmpath([cd,'/_common'])