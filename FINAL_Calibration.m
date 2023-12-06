% Nuttida last updated on 8 Dec 2020:
% Note: 
%   - Compute averages both linear way and circ way
%--------------------------------------------------------------------------

clear all; close all;
scnt = 0;
numsess = 4; % 4 sessions/subj
sub = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]; 
%sub = [11]; % 12082020: sample sub to make a figure
frm = 400;

for s = 1:size(sub, 2) % subject #
    s
    scnt = scnt+1;
    clear pcat in_file out_file
    
    cd (['data/subj' num2str(sub(s))]);
    
    if sub(s) < 10
        in_file = dir(['AxeJEEG_Subj0' num2str(sub(s)) '*Cali1*']);
        
        % Set up a name for an output file
        out_file = ['AxeJEEG_Calib_Subj0' num2str(sub(s)) '.mat'];
    elseif sub(s) > 9
        in_file = dir(['AxeJEEG_Subj' num2str(sub(s)) '*Cali1*']);
        
        % Output file
        out_file = ['AxeJEEG_Calib_Subj' num2str(sub(s)) '.mat'];
    end
    
    label = {'stimDir', 'stimDirREAL', 'joyx', 'joyy'};
    %stimDir = tag assigned to each direction (1-7)
    %stimDirReal = actual direction in rads
    %joyx/joyy = x/y coordinate of the joystick
    
    % Loop through all runs
    for r = 1:size(in_file, 1)
        clear p
        load(in_file(r).name);
        
        % Loop through all 120 trials of calibration
        for t = 1: p.tnumb
            % Distance of the joystick movement
            d(:, t) = sqrt(p.joyx(t,:).^2 + p.joyy(t,:).^2);
            % Find index for max distance
            [~,ind(t)] = max(d(:, t));
            
            % Find (x,y) at max distance
            x_raw(t) = p.joyx(t, ind(t));
            y_raw(t) = p.joyy(t, ind(t));
            
            % Correct for the joysitck's square base
            distc = 1; % corrected joystick distance: 1 au
            rampup_ind = 1:ind(t);
            [c index(t)] = min(abs(d(rampup_ind, t) - distc)); % use this instead to make sure it happens before the peak
            
            % Find (x,y) at max corrected distance which is ~ 1 au
            x(1, t) = p.joyx(t, index(t));
            y(1, t) = p.joyy(t, index(t));
            
            if index(t) <= frm % response was made before deadline
             %111918
             % if ind(t) <= frm; %try defining late responses in a different way
                x(2, t) = p.joyx(t, index(t)); %2nd row says nan for late responses
                y(2, t) = p.joyy(t, index(t));
                
            else % Late response
                x(2, t) = NaN;
                y(2, t) = NaN;
            end
        end
        
        % Find angle of joystick trajectory based on (x,y) at max distance
        [ang, disp] = cart2pol(x,y); %[theta, rho] = cart2pol(x,y); disp = displacement = rho
        angles_all(r, :) = wrapToPi(ang(1, :)); %[-pi pi]; doesn't care about trials with late responses
        angles(r, :) = wrapToPi(ang(2, :)); %trials with late responses have ang set to nan
        index_angles(r, :) = index; 
        
        %%old
        %[ang, ~] = (cart2pol(x,y));
        %ang = wrapToPi(ang); %need to wrap to [-pi pi] so it can later be fed to circ_mean
        
        %reshape things
        p.stimDir = p.stimDir';
        p.joyx = p.joyx';
        p.joyy = p.joyy';
        
        %concatenate
        if r == 1
            for ii = 1:size(label, 2)
                pcat.(label{ii}) = p.(label{ii});
            end
            %concatenate angles of joystick trajectory
            pcat.angles_all = angles_all(r, :); %for calibration, we don't care so much if subjects made 'late' responses;
                                     %otherwise, we'll just be using
                                     %'angles' rather than 'angles_all'
            pcat.index_angles = index_angles(r, :);
        else
            catindex = [2 2 2 2 2 2 2 2 2];
            for ii = 1:size(label, 2)
                pcat.(label{ii}) = cat(catindex(ii), pcat.(label{ii}), p.(label{ii}));
            end
            pcat.angles_all = cat(2, pcat.angles_all, angles_all(r, :));
            pcat.index_angles = cat(2, pcat.index_angles, index_angles(r, :));   
        end
    end
    
    cali.presentedOri = deg2rad([159 123 87 51 15]); % presented orientations; this is in the right order as per the experiment script

    % Loop through all 5 directions
    for j = 1:size(cali.presentedOri, 2)
        cali.meanAngle_circ(j) = circ_mean(pcat.angles_all(pcat.stimDir==j)'); %mean
        cali.medianAngle_circ(j) = circ_median(pcat.angles_all(pcat.stimDir==j)'); %median
        cali.stdAngle_circ(j) = circ_std(pcat.angles_all(pcat.stimDir==j)'); %std
         % not real circular space because we're only using quadrant 1 & 2
        cali.meanAngle(j) = mean(pcat.angles_all(pcat.stimDir==j)'); % mean
        cali.medianAngle(j) = median(pcat.angles_all(pcat.stimDir==j)'); % median
        cali.stdAngle(j) = std(pcat.angles_all(pcat.stimDir==j)'); % std
    end
    
    %----------------------------------------------------------------------
    %% Plot responses from all trials
    %----------------------------------------------------------------------
    cmap = colormap(parula(size(cali.presentedOri, 2)));
 
    figure(s);
    suptitle(['\fontsize{20} Calibration (Subj', num2str(sub(s)) ')'])
    
    subplot(2, 3, 1)
    polar(0, 3, 'w')
    markerSz = 20;
    markerSz2 = 8;
    hold on
    
    % Response directions (for all sessions)
    A1 = polar(pcat.angles_all(pcat.stimDir==5), repmat(2.5, 1, numsess*24) , '.'); hold on;
    A2 = polar(pcat.angles_all(pcat.stimDir==4), repmat(3, 1, numsess*24), '.'); hold on;
    A3 = polar(pcat.angles_all(pcat.stimDir==3), repmat(3.5, 1, numsess*24), '.'); hold on;
    A4 = polar(pcat.angles_all(pcat.stimDir==2), repmat(4, 1, numsess*24), '.'); hold on;
    A5 = polar(pcat.angles_all(pcat.stimDir==1), repmat(4.5, 1, numsess*24), '.'); hold on;
    
    set(A1, 'MarkerSize', markerSz, 'LineWidth', 3, 'color', cmap(1, :))
    set(A2, 'MarkerSize', markerSz, 'LineWidth', 3, 'color', cmap(2, :))
    set(A3, 'MarkerSize', markerSz, 'LineWidth', 3, 'color', cmap(3, :))
    set(A4, 'MarkerSize', markerSz, 'LineWidth', 3, 'color', cmap(4, :))
    set(A5, 'MarkerSize', markerSz, 'LineWidth', 3, 'color', cmap(5, :))
    
    % Actual presented directions
    B1 = polar(cali.presentedOri(5), 2.5, 'd'); hold on;
    B2 = polar(cali.presentedOri(4), 3, 'd'); hold on;
    B3 = polar(cali.presentedOri(3), 3.5, 'd'); hold on;
    B4 = polar(cali.presentedOri(2), 4, 'd'); hold on;
    B5 = polar(cali.presentedOri(1), 4.5, 'd'); hold on;
    
    set(B1, 'MarkerSize', markerSz2, 'LineWidth', 3, 'color', cmap(1, :))
    set(B2, 'MarkerSize', markerSz2, 'LineWidth', 3, 'color', cmap(2, :))
    set(B3, 'MarkerSize', markerSz2, 'LineWidth', 3, 'color', cmap(3, :))
    set(B4, 'MarkerSize', markerSz2, 'LineWidth', 3, 'color', cmap(4, :))
    set(B5, 'MarkerSize', markerSz2, 'LineWidth', 3, 'color', cmap(5, :))
    
    hold off
    title(['\fontsize{16} All trials'])
    
    %----------------------------------------------------------------------
    %% Plot median
    %----------------------------------------------------------------------
    subplot(2, 3, 2)
    polar(0, 4, '-k')
    hold on
    
    % Response directions
    A1 = polar(cali.medianAngle(5), 0.5, '.'); hold on;
    A2 = polar(cali.medianAngle(4), 1, '.'); hold on;
    A3 = polar(cali.medianAngle(3), 1.5, '.'); hold on;
    A4 = polar(cali.medianAngle(2), 2, '.'); hold on;
    A5 = polar(cali.medianAngle(1), 2.5, '.'); hold on;
    
    set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
    set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
    set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
    set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
    set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))
    
    % Actual presented directions
    B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
    B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
    B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
    B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
    B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;
    
    set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
    set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
    set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
    set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
    set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))
    
    title(['\fontsize{16} Median'])
    
    %----------------------------------------------------------------------
    %% Plot mean
    %----------------------------------------------------------------------
    subplot(2, 3, 3)
    polar(0, 4, '-k')
    hold on
    
    % Response directions
    A1 = polar(cali.meanAngle(5), 0.5, '.'); hold on;
    A2 = polar(cali.meanAngle(4), 1, '.'); hold on;
    A3 = polar(cali.meanAngle(3), 1.5, '.'); hold on;
    A4 = polar(cali.meanAngle(2), 2, '.'); hold on;
    A5 = polar(cali.meanAngle(1), 2.5, '.'); hold on;
    
    set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
    set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
    set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
    set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
    set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))
    
    % Actual presented directions
    B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
    B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
    B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
    B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
    B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;
    
    set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
    set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
    set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
    set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
    set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))
    
    title(['\fontsize{16} Mean'])
    
    %% plot circ median
    subplot(2,3,5)
    polar(0, 4, '-k')
    hold on
    
    %response directions
    A1 = polar(cali.medianAngle_circ(5), 0.5, '.'); hold on;
    A2 = polar(cali.medianAngle_circ(4), 1, '.'); hold on;
    A3 = polar(cali.medianAngle_circ(3), 1.5, '.'); hold on;
    A4 = polar(cali.medianAngle_circ(2), 2, '.'); hold on;
    A5 = polar(cali.medianAngle_circ(1), 2.5, '.'); hold on;
    
    set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
    set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
    set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
    set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
    set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))
    
    %actual presented directions
    B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
    B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
    B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
    B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
    B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;
    
    set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
    set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
    set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
    set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
    set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))
    
    title(['\fontsize{16} Median (circ)'])
    
    %% plot circ mean
    subplot(2,3,6)
    polar(0, 4, '-k')
    hold on
    
    %response directions
    A1 = polar(cali.meanAngle_circ(5), 0.5, '.'); hold on;
    A2 = polar(cali.meanAngle_circ(4), 1, '.'); hold on;
    A3 = polar(cali.meanAngle_circ(3), 1.5, '.'); hold on;
    A4 = polar(cali.meanAngle_circ(2), 2, '.'); hold on;
    A5 = polar(cali.meanAngle_circ(1), 2.5, '.'); hold on;
    
    set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
    set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
    set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
    set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
    set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))
    
    %actual presented directions
    B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
    B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
    B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
    B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
    B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;
    
    set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
    set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
    set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
    set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
    set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))
    
    title(['\fontsize{16} Mean (circ)'])
    
    %%
    for iii = 1:5
        allmean(scnt, iii) = cali.meanAngle(:, iii);
        allmedian(scnt, iii) = cali.medianAngle(:, iii);
        allstd(scnt, iii) = cali.stdAngle(:, iii);
        %circ
        allmean_circ(scnt, iii) = cali.meanAngle_circ(:, iii);
        allmedian_circ(scnt, iii) = cali.medianAngle_circ(:, iii);
        allstd_circ(scnt, iii) = cali.stdAngle_circ(:, iii);
    end

    %% save the output file
    save([out_file], 'cali')
    cd ../..
end


%--------------------------------------------------------------------------
%% Plot across-subject averages 
%--------------------------------------------------------------------------
allmean2 = wrapToPi(allmean);
allmedian2 = wrapToPi(allmedian);
allstd2 = wrapToPi(allstd);

avgmean_circ = circ_mean(allmean_circ, [], 1);
avgmedian_circ = circ_mean(allmedian_circ);
avgmean = mean(allmean);
avgmedian = mean(allmedian);

for fig = size(sub, 2)+2

%--------------------------------------------------------------------------
%% Plot median
%--------------------------------------------------------------------------
figure(size(sub, 2)+2)
subplot(2, 2, 1)
polar(0, 4, '-k')
hold on

% Response directions
A1 = polar(avgmedian(5), 0.5, '.'); hold on;
A2 = polar(avgmedian(4), 1, '.'); hold on;
A3 = polar(avgmedian(3), 1.5, '.'); hold on;
A4 = polar(avgmedian(2), 2, '.'); hold on;
A5 = polar(avgmedian(1), 2.5, '.'); hold on;

set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))

% Actual presented directions
B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;

set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))

title(['\fontsize{16} Median'])

%--------------------------------------------------------------------------
%% Plot mean
%--------------------------------------------------------------------------
subplot(2, 2, 2)
polar(0, 4, '-k')
hold on

% Response directions
A1 = polar(avgmean(5), 0.5, '.'); hold on;
A2 = polar(avgmean(4), 1, '.'); hold on;
A3 = polar(avgmean(3), 1.5, '.'); hold on;
A4 = polar(avgmean(2), 2, '.'); hold on;
A5 = polar(avgmean(1), 2.5, '.'); hold on;

set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))

% Actual presented directions
B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;

set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))

title(['\fontsize{16} Mean'])

%--------------------------------------------------------------------------
%% Plot circ median
%--------------------------------------------------------------------------
subplot(2, 2, 3)
polar(0, 4, '-k')
hold on

% Response directions
A1 = polar(avgmedian_circ(5), 0.5, '.'); hold on;
A2 = polar(avgmedian_circ(4), 1, '.'); hold on;
A3 = polar(avgmedian_circ(3), 1.5, '.'); hold on;
A4 = polar(avgmedian_circ(2), 2, '.'); hold on;
A5 = polar(avgmedian_circ(1), 2.5, '.'); hold on;

set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))

% Actual presented directions
B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;

set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))

title(['\fontsize{16} Median (circ)'])

%--------------------------------------------------------------------------
%% Plot circ mean
%--------------------------------------------------------------------------
subplot(2, 2, 4)
polar(0, 4, '-k')
hold on

% Response directions
A1 = polar(avgmean_circ(5), 0.5, '.'); hold on;
A2 = polar(avgmean_circ(4), 1, '.'); hold on;
A3 = polar(avgmean_circ(3), 1.5, '.'); hold on;
A4 = polar(avgmean_circ(2), 2, '.'); hold on;
A5 = polar(avgmean_circ(1), 2.5, '.'); hold on;

set(A1, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(1, :))
set(A2, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(2, :))
set(A3, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(3, :))
set(A4, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(4, :))
set(A5, 'MarkerSize', markerSz, 'LineWidth',3, 'color', cmap(5, :))

% Actual presented directions
B1 = polar(cali.presentedOri(5), 0.5, 'd'); hold on;
B2 = polar(cali.presentedOri(4), 1, 'd'); hold on;
B3 = polar(cali.presentedOri(3), 1.5, 'd'); hold on;
B4 = polar(cali.presentedOri(2), 2, 'd'); hold on;
B5 = polar(cali.presentedOri(1), 2.5, 'd'); hold on;

set(B1, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(1, :))
set(B2, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(2, :))
set(B3, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(3, :))
set(B4, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(4, :))
set(B5, 'MarkerSize', markerSz2, 'LineWidth',3, 'color', cmap(5, :))

title(['\fontsize{16} Mean (circ)'])
end
