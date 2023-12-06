% Nuttida last updated on 20 December 2021
% Note:
%   - This script computes response trajectories and response errors over time
%   - Note on some task parameters:
%       - p.f_precuedur; fixation (with nontg): 48:12:96 frms (400-800 ms)
%       - p.f_cuedur; attention cue (with nontg): 72:12:120 frms (600-1000 ms)
%       - p.f_tgdur; tg presentation: 96 frms (800 ms)
%       - p.f_nontgdur: 60 frms (800 ms)
%       - p.f_ITIdur: 80:120 frms (666.67-1000 ms)
%       - p.f_targetonset = when the tg comes up! (= p.f_precuedur+p.f_cuedur)
%--------------------------------------------------------------------------
clear all; close all;

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% What kind of trials do we want to include in the analysis (default: 2-2)
ana = 2; % 1 = only exclude missed trials; 2 = exclude early (resp offset happened even before tg onset) & missed trials
% Num of frames we want to look at (max possible is 500)
numfr = 500;
% How to treat trials with responses in the opposite directions
opp = 2; % 1 = keep; 2 = remove
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% Set outFile name
if ana == 1
    if opp == 1
        outFile = 'behavDat_FINAL_withEarly_withOppo.mat'
    elseif opp == 2
        outFile = 'behavDat_FINAL_withEarly_noOppo.mat'
    end
elseif ana == 2
    if opp == 1
        outFile = 'behavDat_FINAL_noEarly_withOppo.mat'
    elseif opp == 2
        outFile = 'behavDat_FINAL_noEarly_noOppo.mat'
    end
end

% Set up some parameters
allsub = [1 2 3 4 5 6 7 8 9 10 11 12 14]; % list of subjects; incomplete data: subj 13 & 15
refRate = 120;
fq = 1000/refRate; % 1/refresh rate = duration of each presentation frame
totalTr = 2880; % 4 sessions*6 runs/session*120 trials/run;

scnt = 0;
for s = 1:numel(allsub)
    s
    scnt = scnt+1;
    
    %----------------------------------------------------------------------
    %% Compile data
    %----------------------------------------------------------------------
    % Grab data from all 4 sessions of 6 runs each
    cd (['Data/subj' num2str(allsub(s))]);
    clear pcat
    
    if ismember(allsub(s), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        fname = dir(['*Subj0' num2str(allsub(s)) '*Run*']);
    else
        fname = dir(['*Subj' num2str(allsub(s)) '*Run*']);
    end
    
    label = {'attcue','tgColor','moco','stimDir','stimDirREAL','joyX','joyY', ...
        'fdur','expectOri', 'targetonset', 'f_targetonset'};
    
    % Loop through all 24 runs to stack data
    rnumb = size(fname, 1); % total number of runs
    for r = 1: rnumb
        r
        clear p
        load(fname(r).name);
        p.attcue = p.attCue';
        p.tgColor = p.tgColor';
        p.moco = p.tgCoh';
        p.stimDir = p.stimDir';
        p.joyx = p.joyx';
        p.joyy = p.joyy';
        p.fdur = p.fdur';
        p.f_targetonset = p.f_precuedur + p.f_cuedur;
        p.targetonset = p.f_targetonset*fq;
        
        % Define the expected orientation of the current run
        if mod(r, 6) == 0
            p.expectOri = nan(1, p.tnumb);
        else
            p.expectOri = repmat(p.expOri, [1, p.tnumb]);
        end
        
        % Note: size of p.joyx and p.joyy is 500 frames*120 trials &
        % tg presentation lasts 96 frames and nontg presentation is 120
        % frames max
        allfr = size(p.joyx, 1); % we recorded up to 500 frames after target onset
        
        % Crop joystick responses to however many frames we care about
        p.joyX = p.joyx(1:numfr, :); % x-coordinates of the joystick
        p.joyY = p.joyy(1:numfr, :); % y-coordinates of the joystick
        
        % Loop through all trials in this run
        for t = 1: p.tnumb % number of trials/run = 120 trials
            clear d_temp ind_temp  rampup_ind d_diff_temp index_temp
            d(t, :) = sqrt(p.joyX(:, t).^2 + p.joyY(:, t).^2);
            noise_frm = 80; % peak has to happen after these many frams (in case subjs have a really late resp on the previous trial that spills over)
            d_temp = d(t, noise_frm+1:end); % only use data from after noise_frm to determine peaked resp
            [~, ind_temp] = max(d_temp);
            ind(t) = ind_temp + noise_frm; % index for max traj distance from (0,0)
            %[~, ind(t)] = max(d(t, :)); % find index for max traj distance from (0,0)
            
            %--------------------------------------------------------------
            % Normalize response trajectories
            %--------------------------------------------------------------
            % Find (x,y) at max distance on that trial
            x_raw(t) = p.joyx(ind(t), t);
            y_raw(t) = p.joyy(ind(t), t);
            %resp_raw(t) = arctan(x_raw(t), y_raw(t))
            
            % Correct for the joystick's square base
            distc = 1; % corrected joystick distance
            rampup_ind = noise_frm+1:ind(t); % indices for all frames where the traj is ramping up (after 'noise_frm' timepoints)
            d_diff_temp = abs(d(t, rampup_ind) - distc);
            min_temp = min(d_diff_temp);
            min_round_temp = round(min_temp, 2);
            dist_thres = 0.4; % if a joystick traj doesn't go past this cumulative point (out of 1), then say
            if min_temp >= dist_thres
                [c index_temp] = min(abs(d(t, rampup_ind) - distc));
                index(t) = index_temp + noise_frm;
            elseif min_temp < dist_thres
                if min_round_temp < min_temp % use min_round_temp
                    resp_thres = min_round_temp + 0.01;  % this is a threshold to consider what we want to consider a 'peak' of resp traj
                elseif min_round_temp >= min_temp % use min_round_temp-1
                    resp_thres = min_round_temp;
                end
                index_temp = find(d_diff_temp <= resp_thres, 1);
                index(t) = index_temp + noise_frm;
            end
            
            %figure(t);
            %plot(d(t, :)); hold on; % this is joystick trajectory
            %line([ind(t) ind(t)], [0 1], 'Color', 'red'); hold on; line([index(t) index(t)], [0 1], 'Color', 'black');
            
            % Find (x,y) at max corrected distance which is ~1 a.u. for each
            % trial
            x(1, t) = p.joyx(index(t), t); % first row of both x and y doesn't care if the responses happened after resp deadline
            y(1, t) = p.joyy(index(t), t);
            
            % Keep track of late responses (note: this is only applicable if
            % we set 'response deadline' (through numfr) to be less than
            % the total recorded frames (i.e., numfr < 500)
            if index(t) <= numfr % response was made before deadline
                %if ind(t) <= frm; % try defining late responses in a different way
                x(2, t) = p.joyx(index(t), t); % 2nd row says nan for late responses
                y(2, t) = p.joyy(index(t), t);
            else % late response
                x(2, t) = NaN;
                y(2, t) = NaN;
            end
        end
        
        %         % Plot all responses just to see
        %         figure(s*100+r); suptitle(['subj ', num2str(allsub(s)), ' blk ', num2str(r)])
        %         for ti = 1:p.tnumb
        %             subplot(6, 20, ti); plot(d(ti, :), 'b'); title(['tr ', num2str(ti)]); hold on;
        %             line([index(ti), index(ti)], [0, 1], 'Color', 'red'); % draw a line to indicate where the true peak lies
        %             ylim([0 1.2]);
        %         end
        
        % Find response angles in rad based on (x, y) at max distance
        [ang, disp] = cart2pol(x, y); % [theta, rho] = cart2pol(x,y); disp = displacement
        angles_all(r, :) = wrapToPi(ang(1, :)); % [-pi pi]; only use valid resp
        angles(r, :) = wrapToPi(ang(2, :)); % trials with late responses have ang set to nan
        index_angles(r, :) = index; % index for where resp traj peaks on each trial
        
        % Concatenate
        if r == 1 % run 1
            for ii = 1:size(label, 2)
                pcat.(label{ii}) = p.(label{ii});
            end
            pcat.angles_all = angles_all(r, :);
            pcat.angles = angles(r, :);
            pcat.index_angles = index_angles(r, :);
        else % run 2-24
            catindex = [2 2 2 2 2 2 2 2 2 2 2];
            for ii = 1:size(label, 2)
                pcat.(label{ii})  = cat(catindex(ii), pcat.(label{ii}), p.(label{ii}));
            end
            pcat.angles_all = cat(2, pcat.angles_all, angles_all(r, :)); % resp of all trials
            pcat.angles = cat(2, pcat.angles, angles(r, :)); % resp of only not-late trials
            pcat.index_angles = cat(2, pcat.index_angles, index_angles(r, :));
        end
    end
    
    % Change name of pcat.angles to distinguish it from angle resp/frm
    pcat.finalResp_rad = pcat.angles; % final/trial angle responses; in rad
    
    % Grab the calibration data of each subject for calculation of accuracy, etc.
    if allsub(s) < 10
        load(['AxeJEEG_Calib_Subj0' num2str(allsub(s)) '.mat']);
    else
        load(['AxeJEEG_Calib_Subj' num2str(allsub(s)) '.mat']);
    end
    
    % Calibrated angles from calibration sessions; will be later used to
    % compute accuracy (use this instead of the actual presented angles)
    pcat.caliAngle_rad = cali.medianAngle_circ(pcat.stimDir); % in rad
    pcat.caliAngle = wrapTo180(rad2deg(pcat.caliAngle_rad)); % in deg
    pcat.caliStd_rad = cali.stdAngle_circ(pcat.stimDir); % in rad
    pcat.caliStd = wrapTo180(rad2deg(pcat.caliStd_rad)); % in deg
    
    % Distance of joystick response
    pcat.distance0 = sqrt(pcat.joyX.^2 + pcat.joyY.^2);
    
    %----------------------------------------------------------------------
    % Set up some parameters
    %----------------------------------------------------------------------
    % Define epoch timing for target-locked, resp onset-locked, & peaked
    % resp-locked
    tgOnset_before = 70; % frames; = 70*8.3 = 581 ms before (this cuts into the attention cue presentation)
    tgOnset_after = 241; % frames; = 241*8.3 = 2003 ms after (this cuts into the non-tg presentation)
    respOnset_before = 7; % frames; = 7*8.3 = 58.1 ms before
    respOnset_after = 200; % frames; = 200*8.3 = 1660 ms after
    respOffset_before = 91; % frames; = 91*8.3 = 755.3 ms before
    respOffset_after = 31; % frames; = 31*8.3 = 257.3 ms after
    
    % Pre-allocate storage
    pcat.earlyMissed = zeros(1, size(pcat.distance0, 2)); % to keep track of good and bad trials; 0 being valid; 1 = early (false alarm), 2 = no responses made
    pcat.distanceTgLocked = nan(tgOnset_before + tgOnset_after +1, size(pcat.distance0, 2));
    pcat.anDiffTgLocked = nan(tgOnset_before + tgOnset_after +1, size(pcat.distance0, 2));
    pcat.anRespTgLocked = nan(tgOnset_before + tgOnset_after +1, size(pcat.distance0, 2));
    pcat.distanceRespLocked = nan(respOnset_before + respOnset_after +1, size(pcat.distance0, 2));
    pcat.anDiffRespLocked = nan(respOnset_before + respOnset_after +1, size(pcat.distance0, 2));
    pcat.anRespRespLocked = nan(respOnset_before + respOnset_after +1, size(pcat.distance0, 2));
    pcat.distanceOffLocked = nan(respOffset_before + respOffset_after +1, size(pcat.distance0, 2));
    pcat.anDiffOffLocked = nan(respOffset_before + respOffset_after +1, size(pcat.distance0, 2));
    pcat.anRespOffLocked = nan(respOffset_before + respOffset_after +1, size(pcat.distance0, 2));
    pcat.finalResp = nan(1, size(pcat.distance0, 2));
    pcat.finalError = nan(1, size(pcat.distance0, 2));
    pcat.meanVec = nan(1, size(pcat.distance0, 2));
    pcat.accuracy = nan(1, size(pcat.distance0, 2));
    pcat.accuracy_fixed = nan(1, size(pcat.distance0, 2));
    %pcat.lookAtMe = nan(size(pcat.distance0, 2), 6);
    
    %----------------------------------------------------------------------
    %% Loop through trials to compute resp trajectories and errors
    %----------------------------------------------------------------------
    for tt = 1:size(pcat.distance0, 2)
        tt
        
        %------------------------------------------------------------------
        % Find response onset 
        %------------------------------------------------------------------
        clear tmp_distance0 maxd tmp_resp_index0
        tmp_distance0 = pcat.distance0(:, tt); % joystick dist on this whole trial
        noise = 0.04; % threshold used to make sure we detect 'real' response onsets
        figure; plot(tmp_distance0);
        
        % Note that only count resp onset based on any above-noise movement that happened after 'noise_frm
        %' (these many frms)
        tmp_diff = find(diff(tmp_distance0) > noise); % a list of all timepoints with slope above 'noise'
        tmp_diff_down = find(diff(tmp_distance0) < (-1)*noise); % for the down-ramp part of the resp traj; tmp_diff_down(1) is the first timepoint of the ramp-downt part
        %if isempty(find(tmp_diff > noise_frm, 1)) == 1 % if subj didnt make a response, after the 'noise frame' period
         if isempty(find(tmp_diff > noise_frm)) == 1 % if subj didnt make a response, after the 'noise frame' period
            pcat.resp_index0(tt) = nan;
            pcat.resp_index(tt) = nan;
            pcat.finalResp_frm(tt) = pcat.index_angles(tt);
        else % if a valid resp (i.e., amplitude > noise thresh) was made
            pcat.resp_index0(tt) = tmp_diff(find(tmp_diff > noise_frm, 1)); % frame index for response onset
         end
        
        %------------------------------------------------------------------
        % Peaked resp/resp offset (i.e., the frame of max joystick dist)
        %------------------------------------------------------------------
        if pcat.index_angles(tt) > pcat.resp_index0(tt) % if the frm of max joy dist > onset frm --> good!
            pcat.finalResp_frm(tt) = pcat.index_angles(tt);
        
        elseif pcat.index_angles(tt) <= pcat.resp_index0(tt) % if the frm of max joy dist <= onset frm, then recalculate finalResp_frm
            rampup_ind2 = pcat.resp_index0(tt)+1:numfr; % indices for all frames where the traj is ramping up (after 'noise_frm' timepoints)
            d_diff_temp2 = abs(tmp_distance0(rampup_ind2) - distc);
            min_temp2 = min(d_diff_temp2);
            min_round_temp2 = round(min_temp2, 2);
            dist_thres = 0.4; % if a joystick traj doesn't go past this cumulative point (out of 1), then say
            if min_temp2 >= dist_thres
                [c index_temp2] = min(abs(tmp_distance0(rampup_ind) - distc));
                index2(t) = index_temp2 + pcat.resp_index0(tt);
            elseif min_temp2 < dist_thres
                if min_round_temp2 < min_temp2 % use min_round_temp
                    resp_thres2 = min_round_temp2 + 0.01;  % this is a threshold to consider what we want to consider a 'peak' odf resp traj
                elseif min_round_temp2 >= min_temp2 % use min_round_temp-1
                    resp_thres2 = min_round_temp2;
                end
                index_temp2 = find(d_diff_temp2 <= resp_thres2, 1);
                index2(t) = index_temp2 + pcat.resp_index0(tt);
            end
            pcat.finalResp_frm(tt) = index2(t);
        end
        
        %------------------------------------------------------------------
        % Set early resp (false alarms) to nans
        %------------------------------------------------------------------
        % Note: pcat.resp_index0 has the frame index of all trials;
        % pcat.resp_index has frame index of early trials set to nan
        tmp_resp_index0 = pcat.resp_index0(tt);
        if isnan(tmp_resp_index0) % if no response was made (missed)
           pcat.earlyMissed(tt) = 2; % missed trials
           pcat.finalRespOffset_frm(tt) = nan;
        else % if some kind of responses was made
            %if pcat.finalResp_frm(tt) <= pcat.f_targetonset(tt) % if subjs finish responding before the tg showed up
            if ~isempty(tmp_diff_down) % if resp traj has a ramp-down part to it
                pcat.finalRespOffset_frm(tt) = tmp_diff_down(1); % this is the first timepoint of the ramp-down part of the resp traj
                if tmp_diff_down(1) <= pcat.f_targetonset(tt) % if subjs finish responding before the tg showed up
                    pcat.resp_index(tt) = nan;
                    pcat.earlyMissed(tt) = 1; % early responses
                    %elseif pcat.finalResp_frm(tt) > pcat.f_targetonset(tt)
                elseif tmp_diff_down(1) > pcat.f_targetonset(tt)
                    pcat.resp_index(tt) = tmp_resp_index0; % valid responses (and pcat.earlyMissed(tt) = 0)
                    % note that it's okay if subjs 'start' moving joystick
                    % prior to tg onset
                end
            else % if resp trajectories didn't come down e.g., subjs made a super late resp
                pcat.finalRespOffset_frm(tt) = nan;
                pcat.resp_index(tt) = tmp_resp_index0; % valid responses (and pcat.earlyMissed(tt) = 0)
            end
        end
        
        % just for debugging
        %pcat.lookAtMe(tt, :) = [pcat.resp_index0(tt) pcat.f_targetonset(tt) ...
        %    pcat.index_angles(tt) pcat.finalResp_frm(tt) pcat.finalRespOffset_frm(tt) pcat.earlyMissed(tt)];
        %
        
        % Normalize resp to correct for the non-circular base of joystick--
        % Find max distance for scaling purpose
        maxd(tt) = max(pcat.distance0(:, tt));
        % Scale responses by setting max distance on joystick base to be 1
        if maxd(tt) > distc  %if more than 1 then scale
            pcat.distance(:, tt) = pcat.distance0(:, tt)./maxd(tt);
        else % if less than or equal to 1 already then keep it as is
            pcat.distance(:, tt) = pcat.distance0(:, tt);
        end
        
        %------------------------------------------------------------------
        %% Compute response errors for 1) each frame and 2) each trial
        %------------------------------------------------------------------
        % Reported angle response on each frame
        [pcat.an0(:, tt), ~] = cart2pol(pcat.joyX(:, tt), pcat.joyY(:, tt)); % response on each frame
        %pcat.an0_deg(:, tt) = wrapTo360(rad2deg(pcat.an0(:, tt))); figure;
        %plot(pcat.an0_deg);
        clear tmp_frmAngle
        tmp_frmAngle = pcat.an0(:, tt); % reported angle resp of each frame on this trial
        
        % Get mean resultant vector of resp on the current trial
        tmp_frmAngle = tmp_frmAngle(isnan(tmp_frmAngle) == 0); % remove nans
        pcat.meanVec(tt) = circ_r(tmp_frmAngle);
        
        % Compute error of the reported angle resp on each frame
        pcat.an(:, tt) = wrapToPi(pcat.an0(:, tt)); % in rad; resp in [-pi pi]
        pcat.angresp(:, tt) = wrapTo180(rad2deg(pcat.an(:, tt))); % convert resp to deg
        pcat.angerror(:, tt) =  abs(wrapTo180(rad2deg(pcat.an(:, tt) - ...
            repmat(pcat.caliAngle_rad(:, tt), [numfr, 1])))); % in deg
        %pcat.finalResp(tt) = pcat.an(pcat.finalResp_frm(tt), tt); % final/trial reported angle resp computed at max distance
        pcat.finalError(tt) = pcat.angerror(pcat.finalResp_frm(tt), tt); % final/trial resp error; in deg
        pcat.finalResp(tt) = wrapTo180(rad2deg(pcat.finalResp_rad(tt))); % final/trial reported angle; in deg
        
        %------------------------------------------------------------------
        %% Lock resp traj & errors to tg onset, resp onset, & peaked resp
        %------------------------------------------------------------------
        % Note: we are doing this in 2 ways (depending on what we set for
        % variable ana): 1) If ana = 1, only set things from missed trials
        % (pcat.earlyMissed == 2) to nan; 2) If ana = 2, set things from both
        % missed and early trials (pcat.earlyMissed == 1) to nan
        
        if pcat.earlyMissed(tt) == 2 % missed trials
            % a) Locked to target onset
            pcat.distanceTgLocked(:, tt) = nan; % resp trajectories
            pcat.anDiffTgLocked(:, tt) = nan; % resp error
            pcat.anRespTgLocked(:, tt) = nan; % angle response
            pcat.finalError(tt) = nan;
            pcat.finalResp(tt) = nan;
            pcat.meanVec(tt) = nan;
            % b) Locked to resp onset
            pcat.distanceRespLocked(:, tt) = nan;
            pcat.anDiffRespLocked(:, tt) = nan;
            pcat.anRespRespLocked(:, tt) = nan;
            % c) Locked to peaked resp (i.e., resp offset)
            pcat.distanceOffLocked(:, tt) = nan;
            pcat.anDiffOffLocked(:, tt) = nan;
            pcat.anRespOffLocked(:, tt) = nan;
            % d) Trial resp
            pcat.finalError(tt) = nan;
            pcat.finalResp(tt) = nan;
            pcat.meanVec(tt) = nan;
            
        elseif pcat.earlyMissed(tt) == 1 % early trials
            if ana == 1 % early trials included
                % a) Lock to target onset
                pcat.distanceTgLocked(:, tt) = pcat.distance(pcat.f_targetonset(tt) - ...
                    tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
                pcat.anDiffTgLocked(:, tt) = pcat.angerror(pcat.f_targetonset(tt) - ...
                    tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
                pcat.anRespTgLocked(:, tt) = pcat.angresp(pcat.f_targetonset(tt) - ...
                    tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
                % b) Lock to resp onset
                if pcat.resp_index(tt) + respOnset_after <= numfr % if respones end in time
                    pcat.distanceRespLocked(:, tt) = pcat.distance(pcat.resp_index(tt) - ...
                        respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
                    pcat.anDiffRespLocked(:, tt) = pcat.angerror(pcat.resp_index(tt) - ...
                        respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
                    pcat.anRespRespLocked(:, tt) = pcat.angresp(pcat.resp_index(tt) - ...
                        respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
                elseif pcat.resp_index(tt) + respOnset_after > numfr % if response onset happens so late that the ceiling of our resp locked goes beyond the number of frames we care about (numfr)
                    % then just use up to 'numfr' frame
                    pcat.distanceRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                        = pcat.distance(pcat.resp_index(tt) - respOnset_before : numfr, tt);
                    pcat.anDiffRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                        = pcat.angerror(pcat.resp_index(tt) - respOnset_before : numfr, tt);
                    pcat.anRespRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                        = pcat.angresp(pcat.resp_index(tt) - respOnset_before : numfr, tt);
                end
                % c) Lock to peaked resp (i.e., resp offset)
                if pcat.finalResp_frm(tt) <= respOffset_before % if 'responses' end too soon rare but could happen if subjs were still moving joystick back to center from the previous trial
                    pcat.distanceOffLocked(:, tt) = nan;
                    pcat.anDiffOffLocked(:, tt) = nan;
                    pcat.anRespOffLocked(:, tt) = nan;
                elseif pcat.finalResp_frm(tt) + respOffset_after <= numfr % if respones end in time
                    pcat.distanceOffLocked(:, tt) = pcat.distance(pcat.finalResp_frm(tt) - ...
                        respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
                    pcat.anDiffOffLocked(:, tt) = pcat.angerror(pcat.finalResp_frm(tt) - ...
                        respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
                    pcat.anRespOffLocked(:, tt) = pcat.angresp(pcat.finalResp_frm(tt) - ...
                        respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
                elseif pcat.finalResp_frm(tt) + respOffset_after > numfr % if response onset happens so late that the ceiling of our resp locked goes beyond the number of frames we care about (numfr)
                    % then just use up to 'numfr' frame
                    pcat.distanceOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                        respOffset_before : numfr, 2), tt) = pcat.distance(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
                    pcat.anDiffOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                        respOnffset_before : numfr, 2), tt) = pcat.angerror(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
                    pcat.anRespOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                        respOnffset_before : numfr, 2), tt) = pcat.angresp(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
                end
                
            elseif ana == 2 % early trials excluded
                % a) Locked to target onset
                pcat.distanceTgLocked(:, tt) = nan; % resp trajectories
                pcat.anDiffTgLocked(:, tt) = nan; % resp error
                pcat.anRespTgLocked(:, tt) = nan; % angle response
                % b) Locked to resp onset
                pcat.distanceRespLocked(:, tt) = nan;
                pcat.anDiffRespLocked(:, tt) = nan;
                pcat.anRespRespLocked(:, tt) = nan;
                % c) Locked to peaked resp (i.e., resp offset)
                pcat.distanceOffLocked(:, tt) = nan;
                pcat.anDiffOffLocked(:, tt) = nan;
                pcat.anRespOffLocked(:, tt) = nan;
                % d) Trial resp
                pcat.finalError(tt) = nan;
                pcat.finalResp(tt) = nan;
                pcat.meanVec(tt) = nan;
            end
            
        elseif pcat.earlyMissed(tt) == 0 % valid response trials
            % a) Lock to target onset
            pcat.distanceTgLocked(:, tt) = pcat.distance(pcat.f_targetonset(tt) - ...
                tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
            pcat.anDiffTgLocked(:, tt) = pcat.angerror(pcat.f_targetonset(tt) - ...
                tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
            pcat.anRespTgLocked(:, tt) = pcat.angresp(pcat.f_targetonset(tt) - ...
                tgOnset_before : pcat.f_targetonset(tt) + tgOnset_after, tt);
            % b) Lock to resp onset
            if pcat.resp_index(tt) + respOnset_after <= numfr % if respones end in time
                pcat.distanceRespLocked(:, tt) = pcat.distance(pcat.resp_index(tt) - ...
                    respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
                pcat.anDiffRespLocked(:, tt) = pcat.angerror(pcat.resp_index(tt) - ...
                    respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
                pcat.anRespRespLocked(:, tt) = pcat.angresp(pcat.resp_index(tt) - ...
                    respOnset_before : pcat.resp_index(tt) + respOnset_after, tt);
            elseif pcat.resp_index(tt) + respOnset_after > numfr % if response onset happens so late that the ceiling of our resp locked goes beyond the number of frames we care about (numfr)
                % then just use up to 'numfr' frame
                pcat.distanceRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                    = pcat.distance(pcat.resp_index(tt) - respOnset_before : numfr, tt);
                pcat.anDiffRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                    = pcat.angerror(pcat.resp_index(tt) - respOnset_before : numfr, tt);
                pcat.anRespRespLocked(1: size(pcat.resp_index(tt) - respOnset_before : numfr, 2), tt) ...
                    = pcat.angresp(pcat.resp_index(tt) - respOnset_before : numfr, tt);
            end
            % c) Lock to peaked resp (i.e., resp offset)
            if pcat.finalResp_frm(tt) + respOffset_after <= numfr % if respones end in time
                pcat.distanceOffLocked(:, tt) = pcat.distance(pcat.finalResp_frm(tt) - ...
                    respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
                pcat.anDiffOffLocked(:, tt) = pcat.angerror(pcat.finalResp_frm(tt) - ...
                    respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
                pcat.anRespOffLocked(:, tt) = pcat.angresp(pcat.finalResp_frm(tt) - ...
                    respOffset_before : pcat.finalResp_frm(tt) + respOffset_after, tt);
            elseif pcat.finalResp_frm(tt) + respOffset_after > numfr % if response onset happens so late that the ceiling of our resp locked goes beyond the number of frames we care about (numfr)
                % then just use up to 'numfr' frame
                pcat.distanceOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                    respOffset_before : numfr, 2), tt) = pcat.distance(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
                pcat.anDiffOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                    respOnffset_before : numfr, 2), tt) = pcat.angerror(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
                pcat.anRespOffLocked(1: size(pcat.finalResp_frm(tt) - ...
                    respOnffset_before : numfr, 2), tt) = pcat.angresp(pcat.finalResp_frm(tt) - respOffset_before : numfr, tt);
            end
        end
        
        %------------------------------------------------------------------
        % Compute the good ol' performance accuracy
        %------------------------------------------------------------------
        % 1) Accuracy based on some threshold that varies with direction
        accThres = 4; % set threshold for 'correct' resp to be within thres*std
        if isnan(pcat.finalResp(tt)) % pcat.finalResp is set to nan missed trials (always) & conditionally for early trials depending on 'ana'
            pcat.accuracy(tt) = 0;
        else % for other trials, acc is computed based on the trial error
            if pcat.finalError(tt) <= accThres*pcat.caliStd(tt)
                pcat.accuracy(tt) = 1;
            else
                pcat.accuracy(tt) = 0;
            end
        end
        
        %) Accuracy based on a hard threshold that is fixed across
        % directions and subjs
        accThres_fixed = 18; % each direction is 36 deg apart
        if isnan(pcat.finalResp(tt)) % pcat.finalResp is set to nan missed trials (always) & conditionally for early trials depending on 'ana'
            pcat.accuracy_fixed(tt) = 0;
        else % for other trials, acc is computed based on the trial error
            if pcat.finalError(tt) <= accThres_fixed
                pcat.accuracy_fixed(tt) = 1;
            else
                pcat.accuracy_fixed(tt) = 0;
            end
        end
    end
    
    %----------------------------------------------------------------------
    %% Save out 'trial' measures (i.e., one value per trial)
    %----------------------------------------------------------------------
    % Set up some parameters
    pcat.trlabel = 1:totalTr; % accumulated trial label; trial 1-2880
    pcat.trlabel_1st2nd = repmat([ones(1, p.tnumb/2) ones(1, p.tnumb/2)*2], 1, rnumb); % label for 1st/2nd half
    pcat.prior = ones(1, totalTr)*3; % label for exp cond; unexpected = 3
    pcat.prior(find(pcat.stimDir - pcat.expectOri == 0)) = 1; % expected = 1
    pcat.prior(find(isnan(pcat.expectOri) == 1)) = 2; % neutral = 2
    
    %----------------------------------------------------------------------
    % Get rid of responses made in the oppostie directions
    %%---------------------------------------------------------------------
    % Split trials of each cond in 2 bins where the 2nd bin contains
    % responses in the opposite direction and will thus be discarded before
    % resampling
    % Note: we can identify all 'opposite resp' trials at once but
    % the commented code here:
    % 1) will acheive the same goal and
    % 2) can be adapted to split trials into performance
    % bins as well (this is what the script was originally written to do &
    % that's why there are different sections for each cond)
    
    allbin = 1:2;
    opp_thres = 10; % threshold for response in ~oppposite directions in deg (default = 30)
    
    % Identify trials with rersponses made in the opposite direction
    pcat.allnthbin = nan(1, size(pcat.finalError, 2)); % an array containing the nth bin label
    all_opp_ind = find(abs(pcat.finalError) >= 180-opp_thres & abs(pcat.finalError) <= 180);
    pcat.all_nthbin(all_opp_ind) = allbin(end);
    all_perf = [(pcat.trlabel)' (abs(pcat.finalError))']; % use the real trial label & abs resp error
    all_perf_sorted = sortrows(all_perf, 2);
    all_tr_label_sorted = all_perf_sorted(:, 1); % sorted real trial label
    all_abs_err_sorted = all_perf_sorted(:, 2); % sorted absolute error
    
    % Find index of resp in the opposite direction range (threshold defined above)
    all_lastbn_ind = find(all_abs_err_sorted >= 180-opp_thres & all_abs_err_sorted <= 180);
    if isempty(all_lastbn_ind) % if not a single resp was in opposite directions
        all_bin1_trials = all_perf_sorted; % keep all trials for binning
    else
        all_bin1_trials = [all_tr_label_sorted(1: all_lastbn_ind-1) all_abs_err_sorted(1: all_lastbn_ind-1)]; % remaining trials after taking out 'opposite dir' resp
    end
    
    all_binsz = round(size(all_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %all_splits = [quantile(all_remain(:, 2), 1/3), quantile(all_remain(:, 2), 2/3)]; % find the splitting points
    all_bin2_trials = [all_tr_label_sorted(all_lastbn_ind) all_abs_err_sorted(all_lastbn_ind)];
    opp_trials = all_bin2_trials(:, 1); % trials with resp made int he 'opposite' dir & will be discarded
    %
    %     % Get trial indices for each condition---------------------------------
    %     % Expectation
    %     pcat.tr_exp = pcat.trlabel(pcat.prior == 1);
    %     pcat.tr_un = pcat.trlabel(pcat.prior == 3);
    %     pcat.tr_neu = pcat.trlabel(pcat.prior == 2);
    %
    %     % Attention
    %     pcat.tr_foc = pcat.trlabel(pcat.attcue == 1);
    %     pcat.tr_div = pcat.trlabel(pcat.attcue == 2);
    %
    %     % Coherence
    %     pcat.tr_lo = pcat.trlabel(pcat.moco == 1);
    %     pcat.tr_hi = pcat.trlabel(pcat.moco == 2);
    %
    %     % Expectation trials-----------------------------------------------------
    %     pcat.exp_nthbin = nan(1, size(pcat.finalError(pcat.tr_exp), 2)); % an array containing the nth bin label
    %     exp_opp_ind = find(abs(pcat.finalError(pcat.tr_exp)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_exp)) <= 180);
    %     pcat.exp_nthbin(exp_opp_ind) = allbin(end);
    %     exp_perf = [(pcat.tr_exp)' (abs(pcat.finalError(pcat.tr_exp)))']; % use the real trial label & abs resp error
    %     exp_perf_sorted = sortrows(exp_perf, 2);
    %     exp_tr_label_sorted = exp_perf_sorted(:, 1); % sorted real trial label
    %     exp_abs_err_sorted = exp_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     exp_lastbn_ind = find(exp_abs_err_sorted >= 180-opp_thres & exp_abs_err_sorted <= 180);
    %     if isempty(exp_lastbn_ind) % if not a single resp was in opposite directions
    %         exp_bin1_trials = exp_perf_sorted; % keep all trials for binning
    %     else
    %         exp_bin1_trials = [exp_tr_label_sorted(1: exp_lastbn_ind-1) exp_abs_err_sorted(1: exp_lastbn_ind-1)]; % remaining trials after taking out 'opposite dir' resp
    %     end
    %
    %     exp_binsz = round(size(exp_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %     %exp_splits = [quantile(exp_remain(:, 2), 1/3), quantile(exp_remain(:, 2), 2/3)]; %find the splitting points
    %     exp_bin2_trials = [exp_tr_label_sorted(exp_lastbn_ind) exp_abs_err_sorted(exp_lastbn_ind)];
    %
    %     % Unexpected trials-----------------------------------------------------
    %     pcat.un_nthbin = nan(1, size(pcat.finalError(pcat.tr_un), 2));
    %     un_opp_ind = find(abs(pcat.finalError(pcat.tr_un)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_un)) <= 180);
    %     pcat.un_nthbin(un_opp_ind) = allbin(end);
    %     un_perf = [(pcat.tr_un)' (abs(pcat.finalError(pcat.tr_un)))']; % use the real trial label & abs resp error
    %     un_perf_sorted = sortrows(un_perf, 2);
    %     un_tr_label_sorted = un_perf_sorted(:, 1); % sorted trial label
    %     un_abs_err_sorted = un_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     un_lastbn_ind = find(un_abs_err_sorted >= 180-opp_thres & un_abs_err_sorted <= 180);
    %     if isempty(un_lastbn_ind) % if not a single resp was in opposite directions
    %         un_bin1_trials = un_perf_sorted; % keep all trials for binning
    %     else
    %         un_bin1_trials = [un_tr_label_sorted(1: un_lastbn_ind-1) un_abs_err_sorted(1: un_lastbn_ind-1)]; % remaining trials after taking out 'opposite dir' resp
    %     end
    %
    %     un_binsz = round(size(un_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %     %un_splits = [quantile(un_remain(:, 2), 1/3), quantile(un_remain(:, 2), 2/3)]; %find the splitting points
    %     un_bin2_trials = [un_tr_label_sorted(un_lastbn_ind) un_abs_err_sorted(un_lastbn_ind)];
    %
    %     % Focused trials-------------------------------------------------------
    %     pcat.foc_nthbin = nan(1, size(pcat.finalError(pcat.tr_foc), 2)); % a n array containing the nth bin label
    %     foc_opp_ind = find(abs(pcat.finalError(pcat.tr_foc)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_foc)) <= 180);
    %     pcat.foc_nthbin(foc_opp_ind) = allbin(end);
    %     foc_perf = [(pcat.tr_foc)' (abs(pcat.finalError(pcat.tr_foc)))']; % use the real trial label & abs resp error
    %     foc_perf_sorted = sortrows(foc_perf, 2);
    %     foc_tr_label_sorted = foc_perf_sorted(:, 1); % sorted trial label
    %     foc_abs_err_sorted = foc_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     foc_lastbn_ind = find(foc_abs_err_sorted >= 180-opp_thres & foc_abs_err_sorted <= 180);
    %     if isempty(foc_lastbn_ind) % if not a single resp was in opposite directions
    %         foc_bin1_trials = foc_perf_sorted; % keep all trials for binning
    %     else
    %         foc_bin1_trials = [foc_tr_label_sorted(1: foc_lastbn_ind-1) foc_abs_err_sorted(1: foc_lastbn_ind-1)]; % remaining trials after taking out 'opposite dir' resp
    %     end
    %     foc_binsz = round(size(foc_bin1_trials, 1)./(size(allbin, 2)-1)); %estimate size for the remaining bins (everything but last)
    %     %foc_splits = [quantile(foc_remain(:, 2), 1/3), quantile(foc_remain(:, 2), 2/3)]; %find the splitting points
    %     foc_bin2_trials = [foc_tr_label_sorted(foc_lastbn_ind) foc_abs_err_sorted(foc_lastbn_ind)];
    %
    %     % Divided trials-------------------------------------------------------
    %     pcat.div_nthbin = nan(1, size(pcat.finalError(pcat.tr_div), 2)); % an array containing the nth bin label
    %     div_opp_ind = find(abs(pcat.finalError(pcat.tr_div)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_div)) <= 180);
    %     pcat.div_nthbin(div_opp_ind) = allbin(end);
    %     div_perf = [(pcat.tr_div)' (abs(pcat.finalError(pcat.tr_div)))']; % use the real trial label & abs resp error
    %     div_perf_sorted = sortrows(div_perf, 2);
    %     div_tr_label_sorted = div_perf_sorted(:, 1); % sorted trial label
    %     div_abs_err_sorted = div_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     div_lastbn_ind = find(div_abs_err_sorted >= 180-opp_thres & div_abs_err_sorted <= 180);
    %     if isempty(div_lastbn_ind) % if not a single resp was in opposite directions
    %         div_bin1_trials = div_perf_sorted; % keep all trials for binning
    %     else
    %         div_bin1_trials = [div_tr_label_sorted(1: div_lastbn_ind-1) div_abs_err_sorted(1: div_lastbn_ind-1)]; %remaining trials after taking out 'opposite dir' resp
    %     end
    %     div_binsz = round(size(div_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %     %div_splits = [quantile(div_remain(:, 2), 1/3), quantile(div_remain(:, 2), 2/3)]; %find the splitting points
    %     div_bin2_trials = [div_tr_label_sorted(div_lastbn_ind) div_abs_err_sorted(div_lastbn_ind)];
    %
    %     % Low coherence trials-------------------------------------------------
    %     pcat.lo_nthbin = nan(1, size(pcat.finalError(pcat.tr_lo), 2)); % an array containing the nth bin label
    %     lo_opp_ind = find(abs(pcat.finalError(pcat.tr_lo)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_lo)) <= 180);
    %     pcat.lo_nthbin(lo_opp_ind) = allbin(end);
    %     lo_perf = [(pcat.tr_lo)' (abs(pcat.finalError(pcat.tr_lo)))']; % use the real trial label & abs resp error
    %     lo_perf_sorted = sortrows(lo_perf, 2);
    %     lo_tr_label_sorted = lo_perf_sorted(:, 1); % sorted trial label
    %     lo_abs_err_sorted = lo_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     lo_lastbn_ind = find(lo_abs_err_sorted >= 180-opp_thres & lo_abs_err_sorted <= 180);
    %     if isempty(lo_lastbn_ind) % if not a single resp was in opposite directions
    %         lo_bin1_trials = lo_perf_sorted; % keep all trials for binning
    %     else
    %         lo_bin1_trials = [lo_tr_label_sorted(1: lo_lastbn_ind-1) lo_abs_err_sorted(1: lo_lastbn_ind-1)]; %remaining trials after taking out 'opposite dir' resp
    %     end
    %     lo_binsz = round(size(lo_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %     %lo_splits = [quantile(lo_remain(:, 2), 1/3), quantile(lo_remain(:, 2), 2/3)]; %find the splitting points
    %     lo_bin2_trials = [lo_tr_label_sorted(lo_lastbn_ind) lo_abs_err_sorted(lo_lastbn_ind)];
    %
    %     % High coherence trials------------------------------------------------
    %     pcat.hi_nthbin = nan(1, size(pcat.finalError(pcat.tr_hi), 2)); % an array containing the nth bin label
    %     hi_opp_ind = find(abs(pcat.finalError(pcat.tr_hi)) >= 180-opp_thres & abs(pcat.finalError(pcat.tr_hi)) <= 180);
    %     pcat.hi_nthbin(hi_opp_ind) = allbin(end);
    %     hi_perf = [(pcat.tr_hi)' (abs(pcat.finalError(pcat.tr_hi)))']; % use the real trial label & abs resp error
    %     hi_perf_sorted = sortrows(hi_perf, 2);
    %     hi_tr_label_sorted = hi_perf_sorted(:, 1); % sorted trial label
    %     hi_abs_err_sorted = hi_perf_sorted(:, 2); % sorted absolute error
    %
    %     % Find index of resp in the opposite direction range (threshold defined above)
    %     hi_lastbn_ind = find(hi_abs_err_sorted >= 180-opp_thres & hi_abs_err_sorted <= 180);
    %     if isempty(hi_lastbn_ind) % if not a single resp was in opposite directions
    %         hi_bin1_trials = hi_perf_sorted; % keep all trials for binning
    %     else
    %         hi_bin1_trials = [hi_tr_label_sorted(1: hi_lastbn_ind-1) hi_abs_err_sorted(1: hi_lastbn_ind-1)]; %remaining trials after taking out 'opposite dir' resp
    %     end
    %     hi_binsz = round(size(hi_bin1_trials, 1)./(size(allbin, 2)-1)); % estimate size for the remaining bins (everything but last)
    %     %hi_splits = [quantile(hi_remain(:, 2), 1/3), quantile(hi_remain(:, 2), 2/3)]; %find the splitting points
    %     hi_bin2_trials = [hi_tr_label_sorted(hi_lastbn_ind) hi_abs_err_sorted(hi_lastbn_ind)];
    %
    %     % For brevity----------------------------------------------------------
    %     % Expectation
    %     exp_bin1 = exp_bin1_trials;
    %     exp_bin2 = exp_bin2_trials;
    %     un_bin1 = un_bin1_trials;
    %     un_bin2 = un_bin2_trials;
    %
    %     % Attention
    %     foc_bin1 = foc_bin1_trials;
    %     foc_bin2 = foc_bin2_trials;
    %     div_bin1 = div_bin1_trials;
    %     div_bin2 = div_bin2_trials;
    %
    %     % Coherence
    %     lo_bin1 = lo_bin1_trials;
    %     lo_bin2 = lo_bin2_trials;
    %     hi_bin1 = hi_bin1_trials;
    %     hi_bin2 = hi_bin2_trials;
    
    if opp == 2
        % Get rid of the 'opposite'-resp trials
        pcat.trlabel(opp_trials) = nan;
        pcat.trlabel_1st2nd(opp_trials) = nan;
        %pcat.trlabel = pcat.trlabel(~isnan(pcat.trlabel));
    end
    % Save out trial labels of 'opposite', early, and missed trials
    alldat.oppoTrials{scnt} = opp_trials;
    alldat.earlyTrials{scnt} = find(pcat.earlyMissed == 1); % 1 = early
    alldat.missedTrials{scnt} = find(pcat.earlyMissed == 2); % 2 = missed
    
    %----------------------------------------------------------------------
    % Get trial indices for each condition
    %----------------------------------------------------------------------
    % a) Expectation----------------------------------------------------------
    % All trials from each run
    pcat.tr_exp_all = pcat.trlabel(pcat.prior == 1);
    pcat.tr_un_all = pcat.trlabel(pcat.prior == 3);
    pcat.tr_neu_all = pcat.trlabel(pcat.prior == 2);
    pcat.tr_exp = pcat.tr_exp_all(~isnan(pcat.tr_exp_all)); % take out nans
    pcat.tr_un = pcat.tr_un_all(~isnan(pcat.tr_un_all));
    pcat.tr_neu = pcat.tr_neu_all(~isnan(pcat.tr_neu_all));
    
    % First half of each run
    pcat.tr_exp_first = intersect(pcat.tr_exp, find(pcat.trlabel_1st2nd == 1));
    pcat.tr_un_first = intersect(pcat.tr_un, find(pcat.trlabel_1st2nd == 1));
    pcat.tr_neu_first = intersect(pcat.tr_neu, find(pcat.trlabel_1st2nd == 1));
    
    % Second half of each run
    pcat.tr_exp_second = intersect(pcat.tr_exp, find(pcat.trlabel_1st2nd == 2));
    pcat.tr_un_second = intersect(pcat.tr_un, find(pcat.trlabel_1st2nd == 2));
    pcat.tr_neu_second = intersect(pcat.tr_neu, find(pcat.trlabel_1st2nd == 2));
    
    % b) Attention----------------------------------------------------------
    % All trials from each run
    pcat.tr_foc_all = pcat.trlabel(pcat.attcue == 1);
    pcat.tr_div_all = pcat.trlabel(pcat.attcue == 2);
    pcat.tr_foc = pcat.tr_foc_all(~isnan(pcat.tr_foc_all)); % take out nans
    pcat.tr_div = pcat.tr_div_all(~isnan(pcat.tr_div_all));
    
    % First half of each run
    pcat.tr_foc_first = intersect(pcat.tr_foc, find(pcat.trlabel_1st2nd == 1));
    pcat.tr_div_first = intersect(pcat.tr_div, find(pcat.trlabel_1st2nd == 1));
    
    % Second half of each run
    pcat.tr_foc_second = intersect(pcat.tr_foc, find(pcat.trlabel_1st2nd == 2));
    pcat.tr_div_second = intersect(pcat.tr_div, find(pcat.trlabel_1st2nd == 2));
    
    % c) Coherence----------------------------------------------------------
    % All trials from each run
    pcat.tr_lo_all = pcat.trlabel(pcat.moco == 1);
    pcat.tr_hi_all = pcat.trlabel(pcat.moco == 2);
    pcat.tr_lo = pcat.tr_lo_all(~isnan(pcat.tr_lo_all)); % take out nans
    pcat.tr_hi = pcat.tr_hi_all(~isnan(pcat.tr_hi_all));
    
    % First half of each run
    pcat.tr_lo_first = intersect(pcat.tr_lo, find(pcat.trlabel_1st2nd == 1));
    pcat.tr_hi_first = intersect(pcat.tr_hi, find(pcat.trlabel_1st2nd == 1));
    
    % Second half of each run
    pcat.tr_lo_second = intersect(pcat.tr_lo, find(pcat.trlabel_1st2nd == 2));
    pcat.tr_hi_second = intersect(pcat.tr_hi, find(pcat.trlabel_1st2nd == 2));
    
    % d) Expectation x attention----------------------------------------------
    % All trials from each run
    pcat.tr_exp_foc = intersect(pcat.tr_exp, pcat.tr_foc);
    pcat.tr_exp_div = intersect(pcat.tr_exp, pcat.tr_div);
    pcat.tr_un_foc = intersect(pcat.tr_un, pcat.tr_foc);
    pcat.tr_un_div = intersect(pcat.tr_un, pcat.tr_div);
    pcat.tr_neu_foc = intersect(pcat.tr_neu, pcat.tr_foc);
    pcat.tr_neu_div = intersect(pcat.tr_neu, pcat.tr_div);
    
    % First half of each run
    pcat.tr_exp_foc_first = intersect(pcat.tr_exp_first, pcat.tr_foc_first);
    pcat.tr_exp_div_first = intersect(pcat.tr_exp_first, pcat.tr_div_first);
    pcat.tr_un_foc_first = intersect(pcat.tr_un_first, pcat.tr_foc_first);
    pcat.tr_un_div_first = intersect(pcat.tr_un_first, pcat.tr_div_first);
    pcat.tr_neu_foc_first = intersect(pcat.tr_neu_first, pcat.tr_foc_first);
    pcat.tr_neu_div_first = intersect(pcat.tr_neu_first, pcat.tr_div_first);
    
    % Second half of each run
    pcat.tr_exp_foc_second = intersect(pcat.tr_exp_second, pcat.tr_foc_second);
    pcat.tr_exp_div_second = intersect(pcat.tr_exp_second, pcat.tr_div_second);
    pcat.tr_un_foc_second = intersect(pcat.tr_un_second, pcat.tr_foc_second);
    pcat.tr_un_div_second = intersect(pcat.tr_un_second, pcat.tr_div_second);
    pcat.tr_neu_foc_second = intersect(pcat.tr_neu_second, pcat.tr_foc_second);
    pcat.tr_neu_div_second = intersect(pcat.tr_neu_second, pcat.tr_div_second);
    
    % e) Expectation x coherence----------------------------------------------
    % All trials from each run
    pcat.tr_exp_lo = intersect(pcat.tr_exp, pcat.tr_lo);
    pcat.tr_exp_hi = intersect(pcat.tr_exp, pcat.tr_hi);
    pcat.tr_un_lo = intersect(pcat.tr_un, pcat.tr_lo);
    pcat.tr_un_hi = intersect(pcat.tr_un, pcat.tr_hi);
    pcat.tr_neu_lo = intersect(pcat.tr_neu, pcat.tr_lo);
    pcat.tr_neu_hi = intersect(pcat.tr_neu, pcat.tr_hi);
    
    % First half of each run
    pcat.tr_exp_lo_first = intersect(pcat.tr_exp_first, pcat.tr_lo_first);
    pcat.tr_exp_hi_first = intersect(pcat.tr_exp_first, pcat.tr_hi_first);
    pcat.tr_un_lo_first = intersect(pcat.tr_un_first, pcat.tr_lo_first);
    pcat.tr_un_hi_first = intersect(pcat.tr_un_first, pcat.tr_hi_first);
    pcat.tr_neu_lo_first = intersect(pcat.tr_neu_first, pcat.tr_lo_first);
    pcat.tr_neu_hi_first = intersect(pcat.tr_neu_first, pcat.tr_hi_first);
    
    % Second half of each run
    pcat.tr_exp_lo_second = intersect(pcat.tr_exp_second, pcat.tr_lo_second);
    pcat.tr_exp_hi_second = intersect(pcat.tr_exp_second, pcat.tr_hi_second);
    pcat.tr_un_lo_second = intersect(pcat.tr_un_second, pcat.tr_lo_second);
    pcat.tr_un_hi_second = intersect(pcat.tr_un_second, pcat.tr_hi_second);
    pcat.tr_neu_lo_second = intersect(pcat.tr_neu_second, pcat.tr_lo_second);
    pcat.tr_neu_hi_second = intersect(pcat.tr_neu_second, pcat.tr_hi_second);
    
    % f) Attention x coherence----------------------------------------------
    % All trials from each run
    pcat.tr_foc_lo = intersect(pcat.tr_foc, pcat.tr_lo);
    pcat.tr_foc_hi = intersect(pcat.tr_foc, pcat.tr_hi);
    pcat.tr_div_lo = intersect(pcat.tr_div, pcat.tr_lo);
    pcat.tr_div_hi = intersect(pcat.tr_div, pcat.tr_hi);
    
    % First half of each run
    pcat.tr_foc_lo_first = intersect(pcat.tr_foc_first, pcat.tr_lo_first);
    pcat.tr_foc_hi_first = intersect(pcat.tr_foc_first, pcat.tr_hi_first);
    pcat.tr_div_lo_first = intersect(pcat.tr_div_first, pcat.tr_lo_first);
    pcat.tr_div_hi_first = intersect(pcat.tr_div_first, pcat.tr_hi_first);
    
    % Second half of each run
    pcat.tr_foc_lo_second = intersect(pcat.tr_foc_second, pcat.tr_lo_second);
    pcat.tr_foc_hi_second = intersect(pcat.tr_foc_second, pcat.tr_hi_second);
    pcat.tr_div_lo_second = intersect(pcat.tr_div_second, pcat.tr_lo_second);
    pcat.tr_div_hi_second = intersect(pcat.tr_div_second, pcat.tr_hi_second);
    
    %----------------------------------------------------------------------
    % 1) Final/trial response error
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.finalError_exp{scnt} = pcat.finalError(pcat.tr_exp);
    alldat.finalError_un{scnt} = pcat.finalError(pcat.tr_un);
    alldat.finalError_neu{scnt} = pcat.finalError(pcat.tr_neu);
    first.finalError_exp{scnt} = pcat.finalError(pcat.tr_exp_first);
    first.finalError_un{scnt} = pcat.finalError(pcat.tr_un_first);
    first.finalError_neu{scnt} = pcat.finalError(pcat.tr_neu_first);
    second.finalError_exp{scnt} = pcat.finalError(pcat.tr_exp_second);
    second.finalError_un{scnt} = pcat.finalError(pcat.tr_un_second);
    second.finalError_neu{scnt} = pcat.finalError(pcat.tr_neu_second);
    
    % b) Attention
    alldat.finalError_foc{scnt} = pcat.finalError(pcat.tr_foc);
    alldat.finalError_div{scnt} = pcat.finalError(pcat.tr_div);
    first.finalError_foc{scnt} = pcat.finalError(pcat.tr_foc_first);
    first.finalError_div{scnt} = pcat.finalError(pcat.tr_div_first);
    second.finalError_foc{scnt} = pcat.finalError(pcat.tr_foc_second);
    second.finalError_div{scnt} = pcat.finalError(pcat.tr_div_second);
    
    % c) Coherence
    alldat.finalError_lo{scnt} = pcat.finalError(pcat.tr_lo); % note that this is different from AxeHC
    alldat.finalError_hi{scnt} = pcat.finalError(pcat.tr_hi);
    first.finalError_lo{scnt} = pcat.finalError(pcat.tr_lo_first); % note that this is different from AxeHC
    first.finalError_hi{scnt} = pcat.finalError(pcat.tr_hi_first);
    second.finalError_lo{scnt} = pcat.finalError(pcat.tr_lo_second); % note that this is different from AxeHC
    second.finalError_hi{scnt} = pcat.finalError(pcat.tr_hi_second);
    
    % d) Expectation x attention
    alldat.finalError_exp_foc{scnt} = pcat.finalError(pcat.tr_exp_foc);
    alldat.finalError_exp_div{scnt} = pcat.finalError(pcat.tr_exp_div);
    alldat.finalError_un_foc{scnt} = pcat.finalError(pcat.tr_un_foc);
    alldat.finalError_un_div{scnt} = pcat.finalError(pcat.tr_un_div);
    alldat.finalError_neu_foc{scnt} = pcat.finalError(pcat.tr_neu_foc);
    alldat.finalError_neu_div{scnt} = pcat.finalError(pcat.tr_neu_div);
    
    first.finalError_exp_foc{scnt} = pcat.finalError(pcat.tr_exp_foc_first);
    first.finalError_exp_div{scnt} = pcat.finalError(pcat.tr_exp_div_first);
    first.finalError_un_foc{scnt} = pcat.finalError(pcat.tr_un_foc_first);
    first.finalError_un_div{scnt} = pcat.finalError(pcat.tr_un_div_first);
    first.finalError_neu_foc{scnt} = pcat.finalError(pcat.tr_neu_foc_first);
    first.finalError_neu_div{scnt} = pcat.finalError(pcat.tr_neu_div_first);
    
    second.finalError_exp_foc{scnt} = pcat.finalError(pcat.tr_exp_foc_second);
    second.finalError_exp_div{scnt} = pcat.finalError(pcat.tr_exp_div_second);
    second.finalError_un_foc{scnt} = pcat.finalError(pcat.tr_un_foc_second);
    second.finalError_un_div{scnt} = pcat.finalError(pcat.tr_un_div_second);
    second.finalError_neu_foc{scnt} = pcat.finalError(pcat.tr_neu_foc_second);
    second.finalError_neu_div{scnt} = pcat.finalError(pcat.tr_neu_div_second);
    
    % e) Expectation x coherence
    alldat.finalError_exp_lo{scnt} = pcat.finalError(pcat.tr_exp_lo);
    alldat.finalError_exp_hi{scnt} = pcat.finalError(pcat.tr_exp_hi);
    alldat.finalError_un_lo{scnt} = pcat.finalError(pcat.tr_un_lo);
    alldat.finalError_un_hi{scnt} = pcat.finalError(pcat.tr_un_hi);
    alldat.finalError_neu_lo{scnt} = pcat.finalError(pcat.tr_neu_lo);
    alldat.finalError_neu_hi{scnt} = pcat.finalError(pcat.tr_neu_hi);
    
    first.finalError_exp_lo{scnt} = pcat.finalError(pcat.tr_exp_lo_first);
    first.finalError_exp_hi{scnt} = pcat.finalError(pcat.tr_exp_hi_first);
    first.finalError_un_lo{scnt} = pcat.finalError(pcat.tr_un_lo_first);
    first.finalError_un_hi{scnt} = pcat.finalError(pcat.tr_un_hi_first);
    first.finalError_neu_lo{scnt} = pcat.finalError(pcat.tr_neu_lo_first);
    first.finalError_neu_hi{scnt} = pcat.finalError(pcat.tr_neu_hi_first);
    
    second.finalError_exp_lo{scnt} = pcat.finalError(pcat.tr_exp_lo_second);
    second.finalError_exp_hi{scnt} = pcat.finalError(pcat.tr_exp_hi_second);
    second.finalError_un_lo{scnt} = pcat.finalError(pcat.tr_un_lo_second);
    second.finalError_un_hi{scnt} = pcat.finalError(pcat.tr_un_hi_second);
    second.finalError_neu_lo{scnt} = pcat.finalError(pcat.tr_neu_lo_second);
    second.finalError_neu_hi{scnt} = pcat.finalError(pcat.tr_neu_hi_second);
    
    % f) Attention x coherence
    alldat.finalError_foc_lo{scnt} = pcat.finalError(pcat.tr_foc_lo);
    alldat.finalError_foc_hi{scnt} = pcat.finalError(pcat.tr_foc_hi);
    alldat.finalError_div_lo{scnt} = pcat.finalError(pcat.tr_div_lo);
    alldat.finalError_div_hi{scnt} = pcat.finalError(pcat.tr_div_hi);
    
    first.finalError_foc_lo{scnt} = pcat.finalError(pcat.tr_foc_lo_first);
    first.finalError_foc_hi{scnt} = pcat.finalError(pcat.tr_foc_hi_first);
    first.finalError_div_lo{scnt} = pcat.finalError(pcat.tr_div_lo_first);
    first.finalError_div_hi{scnt} = pcat.finalError(pcat.tr_div_hi_first);
    
    second.finalError_foc_lo{scnt} = pcat.finalError(pcat.tr_foc_lo_second);
    second.finalError_foc_hi{scnt} = pcat.finalError(pcat.tr_foc_hi_second);
    second.finalError_div_lo{scnt} = pcat.finalError(pcat.tr_div_lo_second);
    second.finalError_div_hi{scnt} = pcat.finalError(pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 2) Final/trial response
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.finalResp_exp{scnt} = pcat.finalResp(pcat.tr_exp);
    alldat.finalResp_un{scnt} = pcat.finalResp(pcat.tr_un);
    alldat.finalResp_neu{scnt} = pcat.finalResp(pcat.tr_neu);
    first.finalResp_exp{scnt} = pcat.finalResp(pcat.tr_exp_first);
    first.finalResp_un{scnt} = pcat.finalResp(pcat.tr_un_first);
    first.finalResp_neu{scnt} = pcat.finalResp(pcat.tr_neu_first);
    second.finalResp_exp{scnt} = pcat.finalResp(pcat.tr_exp_second);
    second.finalResp_un{scnt} = pcat.finalResp(pcat.tr_un_second);
    second.finalResp_neu{scnt} = pcat.finalResp(pcat.tr_neu_second);
    
    % b) Attention
    alldat.finalResp_foc{scnt} = pcat.finalResp(pcat.tr_foc);
    alldat.finalResp_div{scnt} = pcat.finalResp(pcat.tr_div);
    first.finalResp_foc{scnt} = pcat.finalResp(pcat.tr_foc_first);
    first.finalResp_div{scnt} = pcat.finalResp(pcat.tr_div_first);
    second.finalResp_foc{scnt} = pcat.finalResp(pcat.tr_foc_second);
    second.finalResp_div{scnt} = pcat.finalResp(pcat.tr_div_second);
    
    % c) Coherence
    alldat.finalResp_lo{scnt} = pcat.finalResp(pcat.tr_lo); % note that this is different from AxeHC
    alldat.finalResp_hi{scnt} = pcat.finalResp(pcat.tr_hi);
    first.finalResp_lo{scnt} = pcat.finalResp(pcat.tr_lo_first); % note that this is different from AxeHC
    first.finalResp_hi{scnt} = pcat.finalResp(pcat.tr_hi_first);
    second.finalResp_lo{scnt} = pcat.finalResp(pcat.tr_lo_second); % note that this is different from AxeHC
    second.finalResp_hi{scnt} = pcat.finalResp(pcat.tr_hi_second);
    
    % d) Expectation x attention
    alldat.finalResp_exp_foc{scnt} = pcat.finalResp(pcat.tr_exp_foc);
    alldat.finalResp_exp_div{scnt} = pcat.finalResp(pcat.tr_exp_div);
    alldat.finalResp_un_foc{scnt} = pcat.finalResp(pcat.tr_un_foc);
    alldat.finalResp_un_div{scnt} = pcat.finalResp(pcat.tr_un_div);
    alldat.finalResp_neu_foc{scnt} = pcat.finalResp(pcat.tr_neu_foc);
    alldat.finalResp_neu_div{scnt} = pcat.finalResp(pcat.tr_neu_div);
    
    first.finalResp_exp_foc{scnt} = pcat.finalResp(pcat.tr_exp_foc_first);
    first.finalResp_exp_div{scnt} = pcat.finalResp(pcat.tr_exp_div_first);
    first.finalResp_un_foc{scnt} = pcat.finalResp(pcat.tr_un_foc_first);
    first.finalResp_un_div{scnt} = pcat.finalResp(pcat.tr_un_div_first);
    first.finalResp_neu_foc{scnt} = pcat.finalResp(pcat.tr_neu_foc_first);
    first.finalResp_neu_div{scnt} = pcat.finalResp(pcat.tr_neu_div_first);
    
    second.finalResp_exp_foc{scnt} = pcat.finalResp(pcat.tr_exp_foc_second);
    second.finalResp_exp_div{scnt} = pcat.finalResp(pcat.tr_exp_div_second);
    second.finalResp_un_foc{scnt} = pcat.finalResp(pcat.tr_un_foc_second);
    second.finalResp_un_div{scnt} = pcat.finalResp(pcat.tr_un_div_second);
    second.finalResp_neu_foc{scnt} = pcat.finalResp(pcat.tr_neu_foc_second);
    second.finalResp_neu_div{scnt} = pcat.finalResp(pcat.tr_neu_div_second);
    
    % e) Expectation x coherence
    alldat.finalResp_exp_lo{scnt} = pcat.finalResp(pcat.tr_exp_lo);
    alldat.finalResp_exp_hi{scnt} = pcat.finalResp(pcat.tr_exp_hi);
    alldat.finalResp_un_lo{scnt} = pcat.finalResp(pcat.tr_un_lo);
    alldat.finalResp_un_hi{scnt} = pcat.finalResp(pcat.tr_un_hi);
    alldat.finalResp_neu_lo{scnt} = pcat.finalResp(pcat.tr_neu_lo);
    alldat.finalResp_neu_hi{scnt} = pcat.finalResp(pcat.tr_neu_hi);
    
    first.finalResp_exp_lo{scnt} = pcat.finalResp(pcat.tr_exp_lo_first);
    first.finalResp_exp_hi{scnt} = pcat.finalResp(pcat.tr_exp_hi_first);
    first.finalResp_un_lo{scnt} = pcat.finalResp(pcat.tr_un_lo_first);
    first.finalResp_un_hi{scnt} = pcat.finalResp(pcat.tr_un_hi_first);
    first.finalResp_neu_lo{scnt} = pcat.finalResp(pcat.tr_neu_lo_first);
    first.finalResp_neu_hi{scnt} = pcat.finalResp(pcat.tr_neu_hi_first);
    
    second.finalResp_exp_lo{scnt} = pcat.finalResp(pcat.tr_exp_lo_second);
    second.finalResp_exp_hi{scnt} = pcat.finalResp(pcat.tr_exp_hi_second);
    second.finalResp_un_lo{scnt} = pcat.finalResp(pcat.tr_un_lo_second);
    second.finalResp_un_hi{scnt} = pcat.finalResp(pcat.tr_un_hi_second);
    second.finalResp_neu_lo{scnt} = pcat.finalResp(pcat.tr_neu_lo_second);
    second.finalResp_neu_hi{scnt} = pcat.finalResp(pcat.tr_neu_hi_second);
    
    % f) Attention x coherence
    alldat.finalResp_foc_lo{scnt} = pcat.finalResp(pcat.tr_foc_lo);
    alldat.finalResp_foc_hi{scnt} = pcat.finalResp(pcat.tr_foc_hi);
    alldat.finalResp_div_lo{scnt} = pcat.finalResp(pcat.tr_div_lo);
    alldat.finalResp_div_hi{scnt} = pcat.finalResp(pcat.tr_div_hi);
    
    first.finalResp_foc_lo{scnt} = pcat.finalResp(pcat.tr_foc_lo_first);
    first.finalResp_foc_hi{scnt} = pcat.finalResp(pcat.tr_foc_hi_first);
    first.finalResp_div_lo{scnt} = pcat.finalResp(pcat.tr_div_lo_first);
    first.finalResp_div_hi{scnt} = pcat.finalResp(pcat.tr_div_hi_first);
    
    second.finalResp_foc_lo{scnt} = pcat.finalResp(pcat.tr_foc_lo_second);
    second.finalResp_foc_hi{scnt} = pcat.finalResp(pcat.tr_foc_hi_second);
    second.finalResp_div_lo{scnt} = pcat.finalResp(pcat.tr_div_lo_second);
    second.finalResp_div_hi{scnt} = pcat.finalResp(pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 3) Mean resultant vector of angle responses on each trial
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.meanVec_exp{scnt} = pcat.meanVec(pcat.tr_exp);
    alldat.meanVec_un{scnt} = pcat.meanVec(pcat.tr_un);
    alldat.meanVec_neu{scnt} = pcat.meanVec(pcat.tr_neu);
    first.meanVec_exp{scnt} = pcat.meanVec(pcat.tr_exp_first);
    first.meanVec_un{scnt} = pcat.meanVec(pcat.tr_un_first);
    first.meanVec_neu{scnt} = pcat.meanVec(pcat.tr_neu_first);
    second.meanVec_exp{scnt} = pcat.meanVec(pcat.tr_exp_second);
    second.meanVec_un{scnt} = pcat.meanVec(pcat.tr_un_second);
    second.meanVec_neu{scnt} = pcat.meanVec(pcat.tr_neu_second);
    
    % b) Attention
    alldat.meanVec_foc{scnt} = pcat.meanVec(pcat.tr_foc);
    alldat.meanVec_div{scnt} = pcat.meanVec(pcat.tr_div);
    first.meanVec_foc{scnt} = pcat.meanVec(pcat.tr_foc_first);
    first.meanVec_div{scnt} = pcat.meanVec(pcat.tr_div_first);
    second.meanVec_foc{scnt} = pcat.meanVec(pcat.tr_foc_second);
    second.meanVec_div{scnt} = pcat.meanVec(pcat.tr_div_second);
    
    % c) Coherence
    alldat.meanVec_lo{scnt} = pcat.meanVec(pcat.tr_lo);
    alldat.meanVec_hi{scnt} = pcat.meanVec(pcat.tr_hi);
    first.meanVec_lo{scnt} = pcat.meanVec(pcat.tr_lo_first);
    first.meanVec_hi{scnt} = pcat.meanVec(pcat.tr_hi_first);
    second.meanVec_lo{scnt} = pcat.meanVec(pcat.tr_lo_second);
    second.meanVec_hi{scnt} = pcat.meanVec(pcat.tr_hi_second);
    
    % d) Expectation x attention
    alldat.meanVec_exp_foc{scnt} = pcat.meanVec(pcat.tr_exp_foc);
    alldat.meanVec_exp_div{scnt} = pcat.meanVec(pcat.tr_exp_div);
    alldat.meanVec_un_foc{scnt} = pcat.meanVec(pcat.tr_un_foc);
    alldat.meanVec_un_div{scnt} = pcat.meanVec(pcat.tr_un_div);
    alldat.meanVec_neu_foc{scnt} = pcat.meanVec(pcat.tr_neu_foc);
    alldat.meanVec_neu_div{scnt} = pcat.meanVec(pcat.tr_neu_div);
    
    first.meanVec_exp_foc{scnt} = pcat.meanVec(pcat.tr_exp_foc_first);
    first.meanVec_exp_div{scnt} = pcat.meanVec(pcat.tr_exp_div_first);
    first.meanVec_un_foc{scnt} = pcat.meanVec(pcat.tr_un_foc_first);
    first.meanVec_un_div{scnt} = pcat.meanVec(pcat.tr_un_div_first);
    first.meanVec_neu_foc{scnt} = pcat.meanVec(pcat.tr_neu_foc_first);
    first.meanVec_neu_div{scnt} = pcat.meanVec(pcat.tr_neu_div_first);
    
    second.meanVec_exp_foc{scnt} = pcat.meanVec(pcat.tr_exp_foc_second);
    second.meanVec_exp_div{scnt} = pcat.meanVec(pcat.tr_exp_div_second);
    second.meanVec_un_foc{scnt} = pcat.meanVec(pcat.tr_un_foc_second);
    second.meanVec_un_div{scnt} = pcat.meanVec(pcat.tr_un_div_second);
    second.meanVec_neu_foc{scnt} = pcat.meanVec(pcat.tr_neu_foc_second);
    second.meanVec_neu_div{scnt} = pcat.meanVec(pcat.tr_neu_div_second);
    
    % e) Expectation x coherence
    alldat.meanVec_exp_lo{scnt} = pcat.meanVec(pcat.tr_exp_lo);
    alldat.meanVec_exp_hi{scnt} = pcat.meanVec(pcat.tr_exp_hi);
    alldat.meanVec_un_lo{scnt} = pcat.meanVec(pcat.tr_un_lo);
    alldat.meanVec_un_hi{scnt} = pcat.meanVec(pcat.tr_un_hi);
    alldat.meanVec_neu_lo{scnt} = pcat.meanVec(pcat.tr_neu_lo);
    alldat.meanVec_neu_hi{scnt} = pcat.meanVec(pcat.tr_neu_hi);
    
    first.meanVec_exp_lo{scnt} = pcat.meanVec(pcat.tr_exp_lo_first);
    first.meanVec_exp_hi{scnt} = pcat.meanVec(pcat.tr_exp_hi_first);
    first.meanVec_un_lo{scnt} = pcat.meanVec(pcat.tr_un_lo_first);
    first.meanVec_un_hi{scnt} = pcat.meanVec(pcat.tr_un_hi_first);
    first.meanVec_neu_lo{scnt} = pcat.meanVec(pcat.tr_neu_lo_first);
    first.meanVec_neu_hi{scnt} = pcat.meanVec(pcat.tr_neu_hi_first);
    
    second.meanVec_exp_lo{scnt} = pcat.meanVec(pcat.tr_exp_lo_second);
    second.meanVec_exp_hi{scnt} = pcat.meanVec(pcat.tr_exp_hi_second);
    second.meanVec_un_lo{scnt} = pcat.meanVec(pcat.tr_un_lo_second);
    second.meanVec_un_hi{scnt} = pcat.meanVec(pcat.tr_un_hi_second);
    second.meanVec_neu_lo{scnt} = pcat.meanVec(pcat.tr_neu_lo_second);
    second.meanVec_neu_hi{scnt} = pcat.meanVec(pcat.tr_neu_hi_second);
    
    % f) Attention x coherence
    alldat.meanVec_foc_lo{scnt} = pcat.meanVec(pcat.tr_foc_lo);
    alldat.meanVec_foc_hi{scnt} = pcat.meanVec(pcat.tr_foc_hi);
    alldat.meanVec_div_lo{scnt} = pcat.meanVec(pcat.tr_div_lo);
    alldat.meanVec_div_hi{scnt} = pcat.meanVec(pcat.tr_div_hi);
    
    first.meanVec_foc_lo{scnt} = pcat.meanVec(pcat.tr_foc_lo_first);
    first.meanVec_foc_hi{scnt} = pcat.meanVec(pcat.tr_foc_hi_first);
    first.meanVec_div_lo{scnt} = pcat.meanVec(pcat.tr_div_lo_first);
    first.meanVec_div_hi{scnt} = pcat.meanVec(pcat.tr_div_hi_first);
    
    second.meanVec_foc_lo{scnt} = pcat.meanVec(pcat.tr_foc_lo_second);
    second.meanVec_foc_hi{scnt} = pcat.meanVec(pcat.tr_foc_hi_second);
    second.meanVec_div_lo{scnt} = pcat.meanVec(pcat.tr_div_lo_second);
    second.meanVec_div_hi{scnt} = pcat.meanVec(pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 4) Response onset time on each trial (time taken to initiate resp)
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.RT1_exp{scnt} = (pcat.resp_index(:, pcat.tr_exp)).*fq; % convert from frm number to ms
    alldat.RT1_un{scnt} = (pcat.resp_index(:, pcat.tr_un)).*fq;
    alldat.RT1_neu{scnt} = (pcat.resp_index(:, pcat.tr_neu)).*fq;
    first.RT1_exp{scnt} = (pcat.resp_index(:, pcat.tr_exp_first)).*fq; % convert from frm number to ms
    first.RT1_un{scnt} = (pcat.resp_index(:, pcat.tr_un_first)).*fq;
    first.RT1_neu{scnt} = (pcat.resp_index(:, pcat.tr_neu_first)).*fq;
    second.RT1_exp{scnt} = (pcat.resp_index(:, pcat.tr_exp_second)).*fq; % convert from frm number to ms
    second.RT1_un{scnt} = (pcat.resp_index(:, pcat.tr_un_second)).*fq;
    second.RT1_neu{scnt} = (pcat.resp_index(:, pcat.tr_neu_second)).*fq;
    
    % b) Attention
    alldat.RT1_foc{scnt} = (pcat.resp_index(:, pcat.tr_foc)).*fq; % convert from frm number to ms
    alldat.RT1_div{scnt} = (pcat.resp_index(:, pcat.tr_div)).*fq;
    first.RT1_foc{scnt} = (pcat.resp_index(:, pcat.tr_foc_first)).*fq; % convert from frm number to ms
    first.RT1_div{scnt} = (pcat.resp_index(:, pcat.tr_div_first)).*fq;
    second.RT1_foc{scnt} = (pcat.resp_index(:, pcat.tr_foc_second)).*fq; % convert from frm number to ms
    second.RT1_div{scnt} = (pcat.resp_index(:, pcat.tr_div_second)).*fq;
    
    % c) Coherence
    alldat.RT1_lo{scnt} = (pcat.resp_index(:, pcat.tr_lo)).*fq; % convert from frm number to ms
    alldat.RT1_hi{scnt} = (pcat.resp_index(:, pcat.tr_hi)).*fq;
    first.RT1_lo{scnt} = (pcat.resp_index(:, pcat.tr_lo_first)).*fq; % convert from frm number to ms
    first.RT1_hi{scnt} = (pcat.resp_index(:, pcat.tr_hi_first)).*fq;
    second.RT1_lo{scnt} = (pcat.resp_index(:, pcat.tr_lo_second)).*fq; % convert from frm number to ms
    second.RT1_hi{scnt} = (pcat.resp_index(:, pcat.tr_hi_second)).*fq;
    
    % d) Expectation x attention
    alldat.RT1_exp_foc{scnt} = (pcat.resp_index(:, pcat.tr_exp_foc)).*fq; % convert from frm number to ms
    alldat.RT1_exp_div{scnt} = (pcat.resp_index(:, pcat.tr_exp_div)).*fq;
    alldat.RT1_un_foc{scnt} = (pcat.resp_index(:, pcat.tr_un_foc)).*fq; % convert from frm number to ms
    alldat.RT1_un_div{scnt} = (pcat.resp_index(:, pcat.tr_un_div)).*fq;
    alldat.RT1_neu_foc{scnt} = (pcat.resp_index(:, pcat.tr_neu_foc)).*fq; % convert from frm number to ms
    alldat.RT1_neu_div{scnt} = (pcat.resp_index(:, pcat.tr_neu_div)).*fq;
    
    first.RT1_exp_foc{scnt} = (pcat.resp_index(:, pcat.tr_exp_foc_first)).*fq; % convert from frm number to ms
    first.RT1_exp_div{scnt} = (pcat.resp_index(:, pcat.tr_exp_div_first)).*fq;
    first.RT1_un_foc{scnt} = (pcat.resp_index(:, pcat.tr_un_foc_first)).*fq; % convert from frm number to ms
    first.RT1_un_div{scnt} = (pcat.resp_index(:, pcat.tr_un_div_first)).*fq;
    first.RT1_neu_foc{scnt} = (pcat.resp_index(:, pcat.tr_neu_foc_first)).*fq; % convert from frm number to ms
    first.RT1_neu_div{scnt} = (pcat.resp_index(:, pcat.tr_neu_div_first)).*fq;
    
    second.RT1_exp_foc{scnt} = (pcat.resp_index(:, pcat.tr_exp_foc_second)).*fq; % convert from frm number to ms
    second.RT1_exp_div{scnt} = (pcat.resp_index(:, pcat.tr_exp_div_second)).*fq;
    second.RT1_un_foc{scnt} = (pcat.resp_index(:, pcat.tr_un_foc_second)).*fq; % convert from frm number to ms
    second.RT1_un_div{scnt} = (pcat.resp_index(:, pcat.tr_un_div_second)).*fq;
    second.RT1_neu_foc{scnt} = (pcat.resp_index(:, pcat.tr_neu_foc_second)).*fq; % convert from frm number to ms
    second.RT1_neu_div{scnt} = (pcat.resp_index(:, pcat.tr_neu_div_second)).*fq;
    
    % e) Expectation x coherence
    alldat.RT1_exp_lo{scnt} = (pcat.resp_index(:, pcat.tr_exp_lo)).*fq; % convert from frm number to ms
    alldat.RT1_exp_hi{scnt} = (pcat.resp_index(:, pcat.tr_exp_hi)).*fq;
    alldat.RT1_un_lo{scnt} = (pcat.resp_index(:, pcat.tr_un_lo)).*fq; % convert from frm number to ms
    alldat.RT1_un_hi{scnt} = (pcat.resp_index(:, pcat.tr_un_hi)).*fq;
    alldat.RT1_neu_lo{scnt} = (pcat.resp_index(:, pcat.tr_neu_lo)).*fq; % convert from frm number to ms
    alldat.RT1_neu_hi{scnt} = (pcat.resp_index(:, pcat.tr_neu_hi)).*fq;
    
    first.RT1_exp_lo{scnt} = (pcat.resp_index(:, pcat.tr_exp_lo_first)).*fq; % convert from frm number to ms
    first.RT1_exp_hi{scnt} = (pcat.resp_index(:, pcat.tr_exp_hi_first)).*fq;
    first.RT1_un_lo{scnt} = (pcat.resp_index(:, pcat.tr_un_lo_first)).*fq; % convert from frm number to ms
    first.RT1_un_hi{scnt} = (pcat.resp_index(:, pcat.tr_un_hi_first)).*fq;
    first.RT1_neu_lo{scnt} = (pcat.resp_index(:, pcat.tr_neu_lo_first)).*fq; % convert from frm number to ms
    first.RT1_neu_hi{scnt} = (pcat.resp_index(:, pcat.tr_neu_hi_first)).*fq;
    
    second.RT1_exp_lo{scnt} = (pcat.resp_index(:, pcat.tr_exp_lo_second)).*fq; % convert from frm number to ms
    second.RT1_exp_hi{scnt} = (pcat.resp_index(:, pcat.tr_exp_hi_second)).*fq;
    second.RT1_un_lo{scnt} = (pcat.resp_index(:, pcat.tr_un_lo_second)).*fq; % convert from frm number to ms
    second.RT1_un_hi{scnt} = (pcat.resp_index(:, pcat.tr_un_hi_second)).*fq;
    second.RT1_neu_lo{scnt} = (pcat.resp_index(:, pcat.tr_neu_lo_second)).*fq; % convert from frm number to ms
    second.RT1_neu_hi{scnt} = (pcat.resp_index(:, pcat.tr_neu_hi_second)).*fq;
    
    % f) Attention x coherence
    alldat.RT1_foc_lo{scnt} = (pcat.resp_index(:, pcat.tr_foc_lo)).*fq; % convert from frm number to ms
    alldat.RT1_foc_hi{scnt} = (pcat.resp_index(:, pcat.tr_foc_hi)).*fq;
    alldat.RT1_div_lo{scnt} = (pcat.resp_index(:, pcat.tr_div_lo)).*fq; % convert from frm number to ms
    alldat.RT1_div_hi{scnt} = (pcat.resp_index(:, pcat.tr_div_hi)).*fq;
    
    first.RT1_foc_lo{scnt} = (pcat.resp_index(:, pcat.tr_foc_lo_first)).*fq; % convert from frm number to ms
    first.RT1_foc_hi{scnt} = (pcat.resp_index(:, pcat.tr_foc_hi_first)).*fq;
    first.RT1_div_lo{scnt} = (pcat.resp_index(:, pcat.tr_div_lo_first)).*fq; % convert from frm number to ms
    first.RT1_div_hi{scnt} = (pcat.resp_index(:, pcat.tr_div_hi_first)).*fq;
    
    second.RT1_foc_lo{scnt} = (pcat.resp_index(:, pcat.tr_foc_lo_second)).*fq; % convert from frm number to ms
    second.RT1_foc_hi{scnt} = (pcat.resp_index(:, pcat.tr_foc_hi_second)).*fq;
    second.RT1_div_lo{scnt} = (pcat.resp_index(:, pcat.tr_div_lo_second)).*fq; % convert from frm number to ms
    second.RT1_div_hi{scnt} = (pcat.resp_index(:, pcat.tr_div_hi_second)).*fq;
    
    %----------------------------------------------------------------------
    % 5) Response offset time on each trial (time taken to reach peaked resp)
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.RT2_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp)).*fq; % convert from frm number to ms
    alldat.RT2_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un)).*fq;
    alldat.RT2_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu)).*fq;
    first.RT2_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_first)).*fq; % convert from frm number to ms
    first.RT2_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_first)).*fq;
    first.RT2_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_first)).*fq;
    second.RT2_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_second)).*fq; % convert from frm number to ms
    second.RT2_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_second)).*fq;
    second.RT2_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_second)).*fq;
    
    % b) Attention
    alldat.RT2_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc)).*fq; % convert from frm number to ms
    alldat.RT2_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div)).*fq;
    first.RT2_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_first)).*fq; % convert from frm number to ms
    first.RT2_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_first)).*fq;
    second.RT2_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_second)).*fq; % convert from frm number to ms
    second.RT2_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_second)).*fq;
    
    % c) Coherence
    alldat.RT2_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo)).*fq; % convert from frm number to ms
    alldat.RT2_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi)).*fq;
    first.RT2_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo_first)).*fq; % convert from frm number to ms
    first.RT2_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi_first)).*fq;
    second.RT2_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo_second)).*fq; % convert from frm number to ms
    second.RT2_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi_second)).*fq;
    
    % d) Expectation x attention
    alldat.RT2_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc)).*fq; % convert from frm number to ms
    alldat.RT2_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div)).*fq;
    alldat.RT2_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc)).*fq; % convert from frm number to ms
    alldat.RT2_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div)).*fq;
    alldat.RT2_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc)).*fq; % convert from frm number to ms
    alldat.RT2_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div)).*fq;
    
    first.RT2_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc_first)).*fq; % convert from frm number to ms
    first.RT2_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div_first)).*fq;
    first.RT2_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc_first)).*fq; % convert from frm number to ms
    first.RT2_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div_first)).*fq;
    first.RT2_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc_first)).*fq; % convert from frm number to ms
    first.RT2_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div_first)).*fq;
    
    second.RT2_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc_second)).*fq; % convert from frm number to ms
    second.RT2_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div_second)).*fq;
    second.RT2_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc_second)).*fq; % convert from frm number to ms
    second.RT2_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div_second)).*fq;
    second.RT2_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc_second)).*fq; % convert from frm number to ms
    second.RT2_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div_second)).*fq;
    
    % e) Expectation x coherence
    alldat.RT2_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo)).*fq; % convert from frm number to ms
    alldat.RT2_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi)).*fq;
    alldat.RT2_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo)).*fq; % convert from frm number to ms
    alldat.RT2_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi)).*fq;
    alldat.RT2_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo)).*fq; % convert from frm number to ms
    alldat.RT2_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi)).*fq;
    
    first.RT2_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo_first)).*fq; % convert from frm number to ms
    first.RT2_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi_first)).*fq;
    first.RT2_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo_first)).*fq; % convert from frm number to ms
    first.RT2_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi_first)).*fq;
    first.RT2_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo_first)).*fq; % convert from frm number to ms
    first.RT2_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi_first)).*fq;
    
    second.RT2_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo_second)).*fq; % convert from frm number to ms
    second.RT2_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi_second)).*fq;
    second.RT2_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo_second)).*fq; % convert from frm number to ms
    second.RT2_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi_second)).*fq;
    second.RT2_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo_second)).*fq; % convert from frm number to ms
    second.RT2_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi_second)).*fq;
    
    % f) Attention x coherence
    alldat.RT2_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo)).*fq; % convert from frm number to ms
    alldat.RT2_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi)).*fq;
    alldat.RT2_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo)).*fq; % convert from frm number to ms
    alldat.RT2_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi)).*fq;
    
    first.RT2_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo_first)).*fq; % convert from frm number to ms
    first.RT2_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi_first)).*fq;
    first.RT2_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo_first)).*fq; % convert from frm number to ms
    first.RT2_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi_first)).*fq;
    
    second.RT2_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo_second)).*fq; % convert from frm number to ms
    second.RT2_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi_second)).*fq;
    second.RT2_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo_second)).*fq; % convert from frm number to ms
    second.RT2_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi_second)).*fq;
    
    %--------------------------------------------------------------------------
    % 6) Reaction time on each trial (time taken from initial movement to peak)
    %--------------------------------------------------------------------------
    % a) Expectation
    alldat.RT_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp)).*fq - (pcat.resp_index(:, pcat.tr_exp)).*fq;
    alldat.RT_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un)).*fq - (pcat.resp_index(:, pcat.tr_un)).*fq;
    alldat.RT_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu)).*fq - (pcat.resp_index(:, pcat.tr_neu)).*fq;
    first.RT_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_first)).*fq - (pcat.resp_index(:, pcat.tr_exp_first)).*fq;
    first.RT_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_first)).*fq - (pcat.resp_index(:, pcat.tr_un_first)).*fq;
    first.RT_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_first)).*fq - (pcat.resp_index(:, pcat.tr_neu_first)).*fq;
    second.RT_exp{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_second)).*fq - (pcat.resp_index(:, pcat.tr_exp_second)).*fq;
    second.RT_un{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_second)).*fq - (pcat.resp_index(:, pcat.tr_un_second)).*fq;
    second.RT_neu{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_second)).*fq - (pcat.resp_index(:, pcat.tr_neu_second)).*fq;
    
    % b) Attention
    alldat.RT_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc)).*fq - (pcat.resp_index(:, pcat.tr_foc)).*fq;
    alldat.RT_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div)).*fq - (pcat.resp_index(:, pcat.tr_div)).*fq;
    first.RT_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_first)).*fq - (pcat.resp_index(:, pcat.tr_foc_first)).*fq;
    first.RT_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_first)).*fq - (pcat.resp_index(:, pcat.tr_div_first)).*fq;
    second.RT_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_second)).*fq - (pcat.resp_index(:, pcat.tr_foc_second)).*fq;
    second.RT_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_second)).*fq - (pcat.resp_index(:, pcat.tr_div_second)).*fq;
    
    % c) Coherence
    alldat.RT_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo)).*fq - (pcat.resp_index(:, pcat.tr_lo)).*fq;
    alldat.RT_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi)).*fq - (pcat.resp_index(:, pcat.tr_hi)).*fq;
    first.RT_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_lo_first)).*fq;
    first.RT_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_hi_first)).*fq;
    second.RT_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_lo_second)).*fq;
    second.RT_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_hi_second)).*fq;
    
    % d) Expectation x attention
    alldat.RT_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc)).*fq - (pcat.resp_index(:, pcat.tr_exp_foc)).*fq;
    alldat.RT_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div)).*fq - (pcat.resp_index(:, pcat.tr_exp_div)).*fq;
    alldat.RT_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc)).*fq - (pcat.resp_index(:, pcat.tr_un_foc)).*fq;
    alldat.RT_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div)).*fq - (pcat.resp_index(:, pcat.tr_un_div)).*fq;
    alldat.RT_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc)).*fq - (pcat.resp_index(:, pcat.tr_neu_foc)).*fq;
    alldat.RT_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div)).*fq - (pcat.resp_index(:, pcat.tr_neu_div)).*fq;
    
    first.RT_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc_first)).*fq - (pcat.resp_index(:, pcat.tr_exp_foc_first)).*fq;
    first.RT_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div_first)).*fq - (pcat.resp_index(:, pcat.tr_exp_div_first)).*fq;
    first.RT_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc_first)).*fq - (pcat.resp_index(:, pcat.tr_un_foc_first)).*fq;
    first.RT_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div_first)).*fq - (pcat.resp_index(:, pcat.tr_un_div_first)).*fq;
    first.RT_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc_first)).*fq - (pcat.resp_index(:, pcat.tr_neu_foc_first)).*fq;
    first.RT_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div_first)).*fq - (pcat.resp_index(:, pcat.tr_neu_div_first)).*fq;
    
    second.RT_exp_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_foc_second)).*fq - (pcat.resp_index(:, pcat.tr_exp_foc_second)).*fq;
    second.RT_exp_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_div_second)).*fq - (pcat.resp_index(:, pcat.tr_exp_div_second)).*fq;
    second.RT_un_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_foc_second)).*fq - (pcat.resp_index(:, pcat.tr_un_foc_second)).*fq;
    second.RT_un_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_div_second)).*fq - (pcat.resp_index(:, pcat.tr_un_div_second)).*fq;
    second.RT_neu_foc{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_foc_second)).*fq - (pcat.resp_index(:, pcat.tr_neu_foc_second)).*fq;
    second.RT_neu_div{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_div_second)).*fq - (pcat.resp_index(:, pcat.tr_neu_div_second)).*fq;
    
    % e) Expectation x coherence
    alldat.RT_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo)).*fq - (pcat.resp_index(:, pcat.tr_exp_lo)).*fq;
    alldat.RT_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi)).*fq - (pcat.resp_index(:, pcat.tr_exp_hi)).*fq;
    alldat.RT_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo)).*fq - (pcat.resp_index(:, pcat.tr_un_lo)).*fq;
    alldat.RT_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi)).*fq - (pcat.resp_index(:, pcat.tr_un_hi)).*fq;
    alldat.RT_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo)).*fq - (pcat.resp_index(:, pcat.tr_neu_lo)).*fq;
    alldat.RT_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi)).*fq - (pcat.resp_index(:, pcat.tr_neu_hi)).*fq;
    
    first.RT_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_exp_lo_first)).*fq;
    first.RT_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_exp_hi_first)).*fq;
    first.RT_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_un_lo_first)).*fq;
    first.RT_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_un_hi_first)).*fq;
    first.RT_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_neu_lo_first)).*fq;
    first.RT_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_neu_hi_first)).*fq;
    
    second.RT_exp_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_exp_lo_second)).*fq;
    second.RT_exp_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_exp_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_exp_hi_second)).*fq;
    second.RT_un_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_un_lo_second)).*fq;
    second.RT_un_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_un_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_un_hi_second)).*fq;
    second.RT_neu_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_neu_lo_second)).*fq;
    second.RT_neu_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_neu_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_neu_hi_second)).*fq;
    
    % f) Attention x coherence
    alldat.RT_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo)).*fq - (pcat.resp_index(:, pcat.tr_foc_lo)).*fq;
    alldat.RT_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi)).*fq - (pcat.resp_index(:, pcat.tr_foc_hi)).*fq;
    alldat.RT_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo)).*fq - (pcat.resp_index(:, pcat.tr_div_lo)).*fq;
    alldat.RT_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi)).*fq - (pcat.resp_index(:, pcat.tr_div_hi)).*fq;
    
    first.RT_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_foc_lo_first)).*fq;
    first.RT_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_foc_hi_first)).*fq;
    first.RT_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo_first)).*fq - (pcat.resp_index(:, pcat.tr_div_lo_first)).*fq;
    first.RT_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi_first)).*fq - (pcat.resp_index(:, pcat.tr_div_hi_first)).*fq;
    
    second.RT_foc_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_foc_lo_second)).*fq;
    second.RT_foc_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_foc_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_foc_hi_second)).*fq;
    second.RT_div_lo{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_lo_second)).*fq - (pcat.resp_index(:, pcat.tr_div_lo_second)).*fq;
    second.RT_div_hi{scnt} = (pcat.finalResp_frm(:, pcat.tr_div_hi_second)).*fq - (pcat.resp_index(:, pcat.tr_div_hi_second)).*fq;
    
    %----------------------------------------------------------------------
    % 7a) Final/trial accuracy computed based on subj std
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.accuracy_exp{scnt} = pcat.accuracy(pcat.tr_exp);
    alldat.accuracy_un{scnt} = pcat.accuracy(pcat.tr_un);
    alldat.accuracy_neu{scnt} = pcat.accuracy(pcat.tr_neu);
    first.accuracy_exp{scnt} = pcat.accuracy(pcat.tr_exp_first);
    first.accuracy_un{scnt} = pcat.accuracy(pcat.tr_un_first);
    first.accuracy_neu{scnt} = pcat.accuracy(pcat.tr_neu_first);
    second.accuracy_exp{scnt} = pcat.accuracy(pcat.tr_exp_second);
    second.accuracy_un{scnt} = pcat.accuracy(pcat.tr_un_second);
    second.accuracy_neu{scnt} = pcat.accuracy(pcat.tr_neu_second);
    
    % b) Attention
    alldat.accuracy_foc{scnt} = pcat.accuracy(pcat.tr_foc);
    alldat.accuracy_div{scnt} = pcat.accuracy(pcat.tr_div);
    first.accuracy_foc{scnt} = pcat.accuracy(pcat.tr_foc_first);
    first.accuracy_div{scnt} = pcat.accuracy(pcat.tr_div_first);
    second.accuracy_foc{scnt} = pcat.accuracy(pcat.tr_foc_second);
    second.accuracy_div{scnt} = pcat.accuracy(pcat.tr_div_second);
    
    % c) Coherence
    alldat.accuracy_lo{scnt} = pcat.accuracy(pcat.tr_lo); % note that this is different from AxeHC
    alldat.accuracy_hi{scnt} = pcat.accuracy(pcat.tr_hi);
    first.accuracy_lo{scnt} = pcat.accuracy(pcat.tr_lo_first); % note that this is different from AxeHC
    first.accuracy_hi{scnt} = pcat.accuracy(pcat.tr_hi_first);
    second.accuracy_lo{scnt} = pcat.accuracy(pcat.tr_lo_second); % note that this is different from AxeHC
    second.accuracy_hi{scnt} = pcat.accuracy(pcat.tr_hi_second);
    
    % d) Expectation x attention
    alldat.accuracy_exp_foc{scnt} = pcat.accuracy(pcat.tr_exp_foc);
    alldat.accuracy_exp_div{scnt} = pcat.accuracy(pcat.tr_exp_div);
    alldat.accuracy_un_foc{scnt} = pcat.accuracy(pcat.tr_un_foc);
    alldat.accuracy_un_div{scnt} = pcat.accuracy(pcat.tr_un_div);
    alldat.accuracy_neu_foc{scnt} = pcat.accuracy(pcat.tr_neu_foc);
    alldat.accuracy_neu_div{scnt} = pcat.accuracy(pcat.tr_neu_div);
    
    first.accuracy_exp_foc{scnt} = pcat.accuracy(pcat.tr_exp_foc_first);
    first.accuracy_exp_div{scnt} = pcat.accuracy(pcat.tr_exp_div_first);
    first.accuracy_un_foc{scnt} = pcat.accuracy(pcat.tr_un_foc_first);
    first.accuracy_un_div{scnt} = pcat.accuracy(pcat.tr_un_div_first);
    first.accuracy_neu_foc{scnt} = pcat.accuracy(pcat.tr_neu_foc_first);
    first.accuracy_neu_div{scnt} = pcat.accuracy(pcat.tr_neu_div_first);
    
    second.accuracy_exp_foc{scnt} = pcat.accuracy(pcat.tr_exp_foc_second);
    second.accuracy_exp_div{scnt} = pcat.accuracy(pcat.tr_exp_div_second);
    second.accuracy_un_foc{scnt} = pcat.accuracy(pcat.tr_un_foc_second);
    second.accuracy_un_div{scnt} = pcat.accuracy(pcat.tr_un_div_second);
    second.accuracy_neu_foc{scnt} = pcat.accuracy(pcat.tr_neu_foc_second);
    second.accuracy_neu_div{scnt} = pcat.accuracy(pcat.tr_neu_div_second);
    
    % e) Expectation x coherence
    alldat.accuracy_exp_lo{scnt} = pcat.accuracy(pcat.tr_exp_lo);
    alldat.accuracy_exp_hi{scnt} = pcat.accuracy(pcat.tr_exp_hi);
    alldat.accuracy_un_lo{scnt} = pcat.accuracy(pcat.tr_un_lo);
    alldat.accuracy_un_hi{scnt} = pcat.accuracy(pcat.tr_un_hi);
    alldat.accuracy_neu_lo{scnt} = pcat.accuracy(pcat.tr_neu_lo);
    alldat.accuracy_neu_hi{scnt} = pcat.accuracy(pcat.tr_neu_hi);
    
    first.accuracy_exp_lo{scnt} = pcat.accuracy(pcat.tr_exp_lo_first);
    first.accuracy_exp_hi{scnt} = pcat.accuracy(pcat.tr_exp_hi_first);
    first.accuracy_un_lo{scnt} = pcat.accuracy(pcat.tr_un_lo_first);
    first.accuracy_un_hi{scnt} = pcat.accuracy(pcat.tr_un_hi_first);
    first.accuracy_neu_lo{scnt} = pcat.accuracy(pcat.tr_neu_lo_first);
    first.accuracy_neu_hi{scnt} = pcat.accuracy(pcat.tr_neu_hi_first);
    
    second.accuracy_exp_lo{scnt} = pcat.accuracy(pcat.tr_exp_lo_second);
    second.accuracy_exp_hi{scnt} = pcat.accuracy(pcat.tr_exp_hi_second);
    second.accuracy_un_lo{scnt} = pcat.accuracy(pcat.tr_un_lo_second);
    second.accuracy_un_hi{scnt} = pcat.accuracy(pcat.tr_un_hi_second);
    second.accuracy_neu_lo{scnt} = pcat.accuracy(pcat.tr_neu_lo_second);
    second.accuracy_neu_hi{scnt} = pcat.accuracy(pcat.tr_neu_hi_second);
    
    % f) Attention x coherence
    alldat.accuracy_foc_lo{scnt} = pcat.accuracy(pcat.tr_foc_lo);
    alldat.accuracy_foc_hi{scnt} = pcat.accuracy(pcat.tr_foc_hi);
    alldat.accuracy_div_lo{scnt} = pcat.accuracy(pcat.tr_div_lo);
    alldat.accuracy_div_hi{scnt} = pcat.accuracy(pcat.tr_div_hi);
    
    first.accuracy_foc_lo{scnt} = pcat.accuracy(pcat.tr_foc_lo_first);
    first.accuracy_foc_hi{scnt} = pcat.accuracy(pcat.tr_foc_hi_first);
    first.accuracy_div_lo{scnt} = pcat.accuracy(pcat.tr_div_lo_first);
    first.accuracy_div_hi{scnt} = pcat.accuracy(pcat.tr_div_hi_first);
    
    second.accuracy_foc_lo{scnt} = pcat.accuracy(pcat.tr_foc_lo_second);
    second.accuracy_foc_hi{scnt} = pcat.accuracy(pcat.tr_foc_hi_second);
    second.accuracy_div_lo{scnt} = pcat.accuracy(pcat.tr_div_lo_second);
    second.accuracy_div_hi{scnt} = pcat.accuracy(pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 7b) Final/trial accuracy computed based on a fixed threshold
    %----------------------------------------------------------------------
    % a) Expectation
    alldat.accuracy_fixed_exp{scnt} = pcat.accuracy_fixed(pcat.tr_exp);
    alldat.accuracy_fixed_un{scnt} = pcat.accuracy_fixed(pcat.tr_un);
    alldat.accuracy_fixed_neu{scnt} = pcat.accuracy_fixed(pcat.tr_neu);
    first.accuracy_fixed_exp{scnt} = pcat.accuracy_fixed(pcat.tr_exp_first);
    first.accuracy_fixed_un{scnt} = pcat.accuracy_fixed(pcat.tr_un_first);
    first.accuracy_fixed_neu{scnt} = pcat.accuracy_fixed(pcat.tr_neu_first);
    second.accuracy_fixed_exp{scnt} = pcat.accuracy_fixed(pcat.tr_exp_second);
    second.accuracy_fixed_un{scnt} = pcat.accuracy_fixed(pcat.tr_un_second);
    second.accuracy_fixed_neu{scnt} = pcat.accuracy_fixed(pcat.tr_neu_second);
    
    % b) Attention
    alldat.accuracy_fixed_foc{scnt} = pcat.accuracy_fixed(pcat.tr_foc);
    alldat.accuracy_fixed_div{scnt} = pcat.accuracy_fixed(pcat.tr_div);
    first.accuracy_fixed_foc{scnt} = pcat.accuracy_fixed(pcat.tr_foc_first);
    first.accuracy_fixed_div{scnt} = pcat.accuracy_fixed(pcat.tr_div_first);
    second.accuracy_fixed_foc{scnt} = pcat.accuracy_fixed(pcat.tr_foc_second);
    second.accuracy_fixed_div{scnt} = pcat.accuracy_fixed(pcat.tr_div_second);
    
    % c) Coherence
    alldat.accuracy_fixed_lo{scnt} = pcat.accuracy_fixed(pcat.tr_lo); % note that this is different from AxeHC
    alldat.accuracy_fixed_hi{scnt} = pcat.accuracy_fixed(pcat.tr_hi);
    first.accuracy_fixed_lo{scnt} = pcat.accuracy_fixed(pcat.tr_lo_first); % note that this is different from AxeHC
    first.accuracy_fixed_hi{scnt} = pcat.accuracy_fixed(pcat.tr_hi_first);
    second.accuracy_fixed_lo{scnt} = pcat.accuracy_fixed(pcat.tr_lo_second); % note that this is different from AxeHC
    second.accuracy_fixed_hi{scnt} = pcat.accuracy_fixed(pcat.tr_hi_second);
    
    % d) Expectation x attention
    alldat.accuracy_fixed_exp_foc{scnt} = pcat.accuracy_fixed(pcat.tr_exp_foc);
    alldat.accuracy_fixed_exp_div{scnt} = pcat.accuracy_fixed(pcat.tr_exp_div);
    alldat.accuracy_fixed_un_foc{scnt} = pcat.accuracy_fixed(pcat.tr_un_foc);
    alldat.accuracy_fixed_un_div{scnt} = pcat.accuracy_fixed(pcat.tr_un_div);
    alldat.accuracy_fixed_neu_foc{scnt} = pcat.accuracy_fixed(pcat.tr_neu_foc);
    alldat.accuracy_fixed_neu_div{scnt} = pcat.accuracy_fixed(pcat.tr_neu_div);
    
    first.accuracy_fixed_exp_foc{scnt} = pcat.accuracy_fixed(pcat.tr_exp_foc_first);
    first.accuracy_fixed_exp_div{scnt} = pcat.accuracy_fixed(pcat.tr_exp_div_first);
    first.accuracy_fixed_un_foc{scnt} = pcat.accuracy_fixed(pcat.tr_un_foc_first);
    first.accuracy_fixed_un_div{scnt} = pcat.accuracy_fixed(pcat.tr_un_div_first);
    first.accuracy_fixed_neu_foc{scnt} = pcat.accuracy_fixed(pcat.tr_neu_foc_first);
    first.accuracy_fixed_neu_div{scnt} = pcat.accuracy_fixed(pcat.tr_neu_div_first);
    
    second.accuracy_fixed_exp_foc{scnt} = pcat.accuracy_fixed(pcat.tr_exp_foc_second);
    second.accuracy_fixed_exp_div{scnt} = pcat.accuracy_fixed(pcat.tr_exp_div_second);
    second.accuracy_fixed_un_foc{scnt} = pcat.accuracy_fixed(pcat.tr_un_foc_second);
    second.accuracy_fixed_un_div{scnt} = pcat.accuracy_fixed(pcat.tr_un_div_second);
    second.accuracy_fixed_neu_foc{scnt} = pcat.accuracy_fixed(pcat.tr_neu_foc_second);
    second.accuracy_fixed_neu_div{scnt} = pcat.accuracy_fixed(pcat.tr_neu_div_second);
    
    % e) Expectation x coherence
    alldat.accuracy_fixed_exp_lo{scnt} = pcat.accuracy_fixed(pcat.tr_exp_lo);
    alldat.accuracy_fixed_exp_hi{scnt} = pcat.accuracy_fixed(pcat.tr_exp_hi);
    alldat.accuracy_fixed_un_lo{scnt} = pcat.accuracy_fixed(pcat.tr_un_lo);
    alldat.accuracy_fixed_un_hi{scnt} = pcat.accuracy_fixed(pcat.tr_un_hi);
    alldat.accuracy_fixed_neu_lo{scnt} = pcat.accuracy_fixed(pcat.tr_neu_lo);
    alldat.accuracy_fixed_neu_hi{scnt} = pcat.accuracy_fixed(pcat.tr_neu_hi);
    
    first.accuracy_fixed_exp_lo{scnt} = pcat.accuracy_fixed(pcat.tr_exp_lo_first);
    first.accuracy_fixed_exp_hi{scnt} = pcat.accuracy_fixed(pcat.tr_exp_hi_first);
    first.accuracy_fixed_un_lo{scnt} = pcat.accuracy_fixed(pcat.tr_un_lo_first);
    first.accuracy_fixed_un_hi{scnt} = pcat.accuracy_fixed(pcat.tr_un_hi_first);
    first.accuracy_fixed_neu_lo{scnt} = pcat.accuracy_fixed(pcat.tr_neu_lo_first);
    first.accuracy_fixed_neu_hi{scnt} = pcat.accuracy_fixed(pcat.tr_neu_hi_first);
    
    second.accuracy_fixed_exp_lo{scnt} = pcat.accuracy_fixed(pcat.tr_exp_lo_second);
    second.accuracy_fixed_exp_hi{scnt} = pcat.accuracy_fixed(pcat.tr_exp_hi_second);
    second.accuracy_fixed_un_lo{scnt} = pcat.accuracy_fixed(pcat.tr_un_lo_second);
    second.accuracy_fixed_un_hi{scnt} = pcat.accuracy_fixed(pcat.tr_un_hi_second);
    second.accuracy_fixed_neu_lo{scnt} = pcat.accuracy_fixed(pcat.tr_neu_lo_second);
    second.accuracy_fixed_neu_hi{scnt} = pcat.accuracy_fixed(pcat.tr_neu_hi_second);
    
    % f) Attention x coherence
    alldat.accuracy_fixed_foc_lo{scnt} = pcat.accuracy_fixed(pcat.tr_foc_lo);
    alldat.accuracy_fixed_foc_hi{scnt} = pcat.accuracy_fixed(pcat.tr_foc_hi);
    alldat.accuracy_fixed_div_lo{scnt} = pcat.accuracy_fixed(pcat.tr_div_lo);
    alldat.accuracy_fixed_div_hi{scnt} = pcat.accuracy_fixed(pcat.tr_div_hi);
    
    first.accuracy_fixed_foc_lo{scnt} = pcat.accuracy_fixed(pcat.tr_foc_lo_first);
    first.accuracy_fixed_foc_hi{scnt} = pcat.accuracy_fixed(pcat.tr_foc_hi_first);
    first.accuracy_fixed_div_lo{scnt} = pcat.accuracy_fixed(pcat.tr_div_lo_first);
    first.accuracy_fixed_div_hi{scnt} = pcat.accuracy_fixed(pcat.tr_div_hi_first);
    
    second.accuracy_fixed_foc_lo{scnt} = pcat.accuracy_fixed(pcat.tr_foc_lo_second);
    second.accuracy_fixed_foc_hi{scnt} = pcat.accuracy_fixed(pcat.tr_foc_hi_second);
    second.accuracy_fixed_div_lo{scnt} = pcat.accuracy_fixed(pcat.tr_div_lo_second);
    second.accuracy_fixed_div_hi{scnt} = pcat.accuracy_fixed(pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    %% Save out continuous measures (i.e., many values per trial)
    %----------------------------------------------------------------------
    % Set measures from trials whose resps are in opposite dir to nan
    if opp == 2
        pcat.distanceTgLocked(:, opp_trials) = nan;
        pcat.distanceRespLocked(:, opp_trials) = nan;
        pcat.distanceOffLocked(:, opp_trials) = nan;
        
        pcat.anDiffTglocked(:, opp_trials) = nan;
        pcat.anDiffResplocked(:, opp_trials) = nan;
        pcat.anDiffOfflocked(:, opp_trials) = nan;
        
        pcat.anRespTgLocked(:, opp_trials) = nan;
        pcat.anRespRespLocked(:, opp_trials) = nan;
        pcat.anRespOffLocked(:, opp_trials) = nan;
    end
    
    %----------------------------------------------------------------------
    % 1) Joystick cumulative distance on each frame
    %----------------------------------------------------------------------
    % All trials in each run
    alldat.tg.dist{scnt} = pcat.distanceTgLocked;
    alldat.respOn.dist{scnt} = pcat.distanceRespLocked;
    alldat.respOff.dist{scnt} = pcat.distanceOffLocked;
    
    % Only trials from the first half of each run
    first.tg.dist{scnt} = pcat.distanceTgLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOn.dist{scnt} = pcat.distanceRespLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOff.dist{scnt} = pcat.distanceOffLocked(:, find(pcat.trlabel_1st2nd == 1));
    
    % Only trials from the second half of each run
    second.tg.dist{scnt} = pcat.distanceTgLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOn.dist{scnt} = pcat.distanceRespLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOff.dist{scnt} = pcat.distanceOffLocked(:, find(pcat.trlabel_1st2nd == 2));
    
    % a) Expectation: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.dist_exp{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp);
    alldat.tg.dist_un{scnt} = pcat.distanceTgLocked(:, pcat.tr_un);
    alldat.tg.dist_neu{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu);
    
    alldat.respOn.dist_exp{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp);
    alldat.respOn.dist_un{scnt} = pcat.distanceRespLocked(:, pcat.tr_un);
    alldat.respOn.dist_neu{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu);
    
    alldat.respOff.dist_exp{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp);
    alldat.respOff.dist_un{scnt} = pcat.distanceOffLocked(:, pcat.tr_un);
    alldat.respOff.dist_neu{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu);
    
    % Only trials from the first half of each run
    first.tg.dist_exp{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_first);
    first.tg.dist_un{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_first);
    first.tg.dist_neu{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_first);
    
    first.respOn.dist_exp{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_first);
    first.respOn.dist_un{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_first);
    first.respOn.dist_neu{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_first);
    
    first.respOff.dist_exp{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_first);
    first.respOff.dist_un{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_first);
    first.respOff.dist_neu{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_first);
    
    % Only trials from the first half of each run
    second.tg.dist_exp{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_second);
    second.tg.dist_un{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_second);
    second.tg.dist_neu{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_second);
    
    second.respOn.dist_exp{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_second);
    second.respOn.dist_un{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_second);
    second.respOn.dist_neu{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_second);
    
    second.respOff.dist_exp{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_second);
    second.respOff.dist_un{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_second);
    second.respOff.dist_neu{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_second);
    
    % b) Attention: tg-locked, resp onset-locked, peaked resp-locked----------
    % All trials in each run
    alldat.tg.dist_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc);
    alldat.tg.dist_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_div);
    
    alldat.respOn.dist_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc);
    alldat.respOn.dist_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_div);
    
    alldat.respOff.dist_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc);
    alldat.respOff.dist_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_div);
    
    % Only trials from the first half of each run
    first.tg.dist_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_first);
    first.tg.dist_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_first);
    
    first.respOn.dist_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_first);
    first.respOn.dist_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_first);
    
    first.respOff.dist_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_first);
    first.respOff.dist_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_first);
    
    % Only trials from the second half of each run
    second.tg.dist_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_second);
    second.tg.dist_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_second);
    
    second.respOn.dist_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_second);
    second.respOn.dist_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_second);
    
    second.respOff.dist_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_second);
    second.respOff.dist_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_second);
    
    % c) Coherence: tg-locked, resp onset-locked, peaked resp-locked---------
    % All trials in each run
    alldat.tg.dist_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_lo);
    alldat.tg.dist_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_hi);
    
    alldat.respOn.dist_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_lo);
    alldat.respOn.dist_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_hi);
    
    alldat.respOff.dist_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_lo);
    alldat.respOff.dist_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_hi);
    
    % Only trials from the first half of each run
    first.tg.dist_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_lo_first);
    first.tg.dist_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_hi_first);
    
    first.respOn.dist_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_lo_first);
    first.respOn.dist_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_hi_first);
    
    first.respOff.dist_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_lo_first);
    first.respOff.dist_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_hi_first);
    
    % Only trials from the second half of each run
    second.tg.dist_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_lo_second);
    second.tg.dist_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_hi_second);
    
    second.respOn.dist_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_lo_second);
    second.respOn.dist_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_hi_second);
    
    second.respOff.dist_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_lo_second);
    second.respOff.dist_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_hi_second);
    
    % d) Expectation x attention: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.dist_exp_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_foc);
    alldat.tg.dist_exp_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_div);
    alldat.tg.dist_un_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_foc);
    alldat.tg.dist_un_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_div);
    alldat.tg.dist_neu_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_foc);
    alldat.tg.dist_neu_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_div);
    
    alldat.respOn.dist_exp_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.dist_exp_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.dist_un_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_foc);
    alldat.respOn.dist_un_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_div);
    alldat.respOn.dist_neu_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_foc);
    alldat.respOn.dist_neu_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_div);
    
    alldat.respOff.dist_exp_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_foc);
    alldat.respOff.dist_exp_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_div);
    alldat.respOff.dist_un_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_foc);
    alldat.respOff.dist_un_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_div);
    alldat.respOff.dist_neu_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_foc);
    alldat.respOff.dist_neu_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_div);
    
    % Only trials from the first half of each run
    first.tg.dist_exp_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_foc_first);
    first.tg.dist_exp_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_div_first);
    first.tg.dist_un_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_foc_first);
    first.tg.dist_un_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_div_first);
    first.tg.dist_neu_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_foc_first);
    first.tg.dist_neu_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_div_first);
    
    first.respOn.dist_exp_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.dist_exp_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.dist_un_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_foc_first);
    first.respOn.dist_un_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_div_first);
    first.respOn.dist_neu_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_foc_first);
    first.respOn.dist_neu_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_div_first);
    
    first.respOff.dist_exp_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_foc_first);
    first.respOff.dist_exp_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_div_first);
    first.respOff.dist_un_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_foc_first);
    first.respOff.dist_un_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_div_first);
    first.respOff.dist_neu_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_foc_first);
    first.respOff.dist_neu_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_div_first);
    
    % Only trials from the second half of each run
    second.tg.dist_exp_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_foc_second);
    second.tg.dist_exp_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_div_second);
    second.tg.dist_un_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_foc_second);
    second.tg.dist_un_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_div_second);
    second.tg.dist_neu_foc{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_foc_second);
    second.tg.dist_neu_div{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_div_second);
    
    second.respOn.dist_exp_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.dist_exp_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.dist_un_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_foc_second);
    second.respOn.dist_un_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_div_second);
    second.respOn.dist_neu_foc{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_foc_second);
    second.respOn.dist_neu_div{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_div_second);
    
    second.respOff.dist_exp_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_foc_second);
    second.respOff.dist_exp_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_div_second);
    second.respOff.dist_un_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_foc_second);
    second.respOff.dist_un_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_div_second);
    second.respOff.dist_neu_foc{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_foc_second);
    second.respOff.dist_neu_div{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_div_second);
    
    % e) Expectation x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.dist_exp_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_lo);
    alldat.tg.dist_exp_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_hi);
    alldat.tg.dist_un_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_lo);
    alldat.tg.dist_un_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_hi);
    alldat.tg.dist_neu_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_lo);
    alldat.tg.dist_neu_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_hi);
    
    alldat.respOn.dist_exp_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.dist_exp_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.dist_un_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_lo);
    alldat.respOn.dist_un_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_hi);
    alldat.respOn.dist_neu_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_lo);
    alldat.respOn.dist_neu_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_hi);
    
    alldat.respOff.dist_exp_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_lo);
    alldat.respOff.dist_exp_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_hi);
    alldat.respOff.dist_un_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_lo);
    alldat.respOff.dist_un_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_hi);
    alldat.respOff.dist_neu_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_lo);
    alldat.respOff.dist_neu_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_hi);
    
    % Only trials from the first half of each run
    first.tg.dist_exp_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_lo_first);
    first.tg.dist_exp_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_hi_first);
    first.tg.dist_un_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_lo_first);
    first.tg.dist_un_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_hi_first);
    first.tg.dist_neu_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_lo_first);
    first.tg.dist_neu_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_hi_first);
    
    first.respOn.dist_exp_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.dist_exp_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.dist_un_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_lo_first);
    first.respOn.dist_un_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_hi_first);
    first.respOn.dist_neu_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_lo_first);
    first.respOn.dist_neu_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_hi_first);
    
    first.respOff.dist_exp_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_lo_first);
    first.respOff.dist_exp_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_hi_first);
    first.respOff.dist_un_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_lo_first);
    first.respOff.dist_un_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_hi_first);
    first.respOff.dist_neu_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_lo_first);
    first.respOff.dist_neu_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_hi_first);
    
    % Only trials from the second half of each run
    second.tg.dist_exp_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_lo_second);
    second.tg.dist_exp_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_exp_hi_second);
    second.tg.dist_un_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_lo_second);
    second.tg.dist_un_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_un_hi_second);
    second.tg.dist_neu_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_lo_second);
    second.tg.dist_neu_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_neu_hi_second);
    
    second.respOn.dist_exp_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.dist_exp_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.dist_un_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_lo_second);
    second.respOn.dist_un_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_un_hi_second);
    second.respOn.dist_neu_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_lo_second);
    second.respOn.dist_neu_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_neu_hi_second);
    
    second.respOff.dist_exp_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_lo_second);
    second.respOff.dist_exp_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_exp_hi_second);
    second.respOff.dist_un_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_lo_second);
    second.respOff.dist_un_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_un_hi_second);
    second.respOff.dist_neu_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_lo_second);
    second.respOff.dist_neu_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_neu_hi_second);
    
    % f) Attention x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.dist_foc_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_lo);
    alldat.tg.dist_foc_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_hi);
    alldat.tg.dist_div_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_lo);
    alldat.tg.dist_div_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_hi);
    
    alldat.respOn.dist_foc_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.dist_foc_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.dist_div_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_lo);
    alldat.respOn.dist_div_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_hi);
    
    alldat.respOff.dist_foc_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_lo);
    alldat.respOff.dist_foc_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_hi);
    alldat.respOff.dist_div_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_lo);
    alldat.respOff.dist_div_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_hi);
    
    % Only trials from the first half of each rdiv
    first.tg.dist_foc_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_lo_first);
    first.tg.dist_foc_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_hi_first);
    first.tg.dist_div_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_lo_first);
    first.tg.dist_div_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_hi_first);
    
    first.respOn.dist_foc_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.dist_foc_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.dist_div_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_lo_first);
    first.respOn.dist_div_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_hi_first);
    
    first.respOff.dist_foc_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_lo_first);
    first.respOff.dist_foc_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_hi_first);
    first.respOff.dist_div_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_lo_first);
    first.respOff.dist_div_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_hi_first);
    
    % Only trials from the second half of each rdiv
    second.tg.dist_foc_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_lo_second);
    second.tg.dist_foc_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_foc_hi_second);
    second.tg.dist_div_lo{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_lo_second);
    second.tg.dist_div_hi{scnt} = pcat.distanceTgLocked(:, pcat.tr_div_hi_second);
    
    second.respOn.dist_foc_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.dist_foc_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.dist_div_lo{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_lo_second);
    second.respOn.dist_div_hi{scnt} = pcat.distanceRespLocked(:, pcat.tr_div_hi_second);
    
    second.respOff.dist_foc_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_lo_second);
    second.respOff.dist_foc_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_foc_hi_second);
    second.respOff.dist_div_lo{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_lo_second);
    second.respOff.dist_div_hi{scnt} = pcat.distanceOffLocked(:, pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 2) Response error on each frame
    %----------------------------------------------------------------------
    % All trials in each run
    alldat.tg.err{scnt} = pcat.anDiffTgLocked(:, tt);
    alldat.respOn.err{scnt} = pcat.anDiffRespLocked(:, tt);
    alldat.respOff.err{scnt} = pcat.anDiffOffLocked(:, tt);
    
    % Only trials from the first half of each run
    first.tg.err{scnt} = pcat.anDiffTgLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOn.err{scnt} = pcat.anDiffRespLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOff.err{scnt} = pcat.anDiffOffLocked(:, find(pcat.trlabel_1st2nd == 1));
    
    % Only trials from the second half of each run
    second.tg.err{scnt} = pcat.anDiffTgLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOn.err{scnt} = pcat.anDiffRespLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOff.err{scnt} = pcat.anDiffOffLocked(:, find(pcat.trlabel_1st2nd == 2));
    
    % a) Expectation: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.err_exp{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp);
    alldat.tg.err_un{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un);
    alldat.tg.err_neu{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu);
    
    alldat.respOn.err_exp{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp);
    alldat.respOn.err_un{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un);
    alldat.respOn.err_neu{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu);
    
    alldat.respOff.err_exp{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp);
    alldat.respOff.err_un{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un);
    alldat.respOff.err_neu{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu);
    
    % Only trials from the first half of each run
    first.tg.err_exp{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_first);
    first.tg.err_un{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_first);
    first.tg.err_neu{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_first);
    
    first.respOn.err_exp{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_first);
    first.respOn.err_un{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_first);
    first.respOn.err_neu{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_first);
    
    first.respOff.err_exp{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_first);
    first.respOff.err_un{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_first);
    first.respOff.err_neu{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_first);
    
    % Only trials from the second half of each run
    second.tg.err_exp{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_second);
    second.tg.err_un{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_second);
    second.tg.err_neu{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_second);
    
    second.respOn.err_exp{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_second);
    second.respOn.err_un{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_second);
    second.respOn.err_neu{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_second);
    
    second.respOff.err_exp{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_second);
    second.respOff.err_un{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_second);
    second.respOff.err_neu{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_second);
    
    % b) Attention: tg-locked, resp onset-locked, peaked resp-locked----------
    % All trials in each run
    alldat.tg.err_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc);
    alldat.tg.err_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div);
    
    alldat.respOn.err_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc);
    alldat.respOn.err_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div);
    
    alldat.respOff.err_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc);
    alldat.respOff.err_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div);
    
    % Only trials from the first half of each run
    first.tg.err_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_first);
    first.tg.err_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_first);
    
    first.respOn.err_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_first);
    first.respOn.err_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_first);
    
    first.respOff.err_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_first);
    first.respOff.err_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_first);
    
    % Only trials from the second half of each run
    second.tg.err_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_second);
    second.tg.err_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_second);
    
    second.respOn.err_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_second);
    second.respOn.err_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_second);
    
    second.respOff.err_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_second);
    second.respOff.err_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_second);
    
    % c) Coherence: tg-locked, resp onset-locked, peaked resp-locked----------
    % All trials in each run
    alldat.tg.err_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_lo);
    alldat.tg.err_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_hi);
    
    alldat.respOn.err_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_lo);
    alldat.respOn.err_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_hi);
    
    alldat.respOff.err_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_lo);
    alldat.respOff.err_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_hi);
    
    % Only trials from the first half of each run
    first.tg.err_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_lo_first);
    first.tg.err_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_hi_first);
    
    first.respOn.err_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_lo_first);
    first.respOn.err_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_hi_first);
    
    first.respOff.err_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_lo_first);
    first.respOff.err_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_hi_first);
    
    % Only trials from the second half of each run
    second.tg.err_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_lo_second);
    second.tg.err_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_hi_second);
    
    second.respOn.err_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_lo_second);
    second.respOn.err_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_hi_second);
    
    second.respOff.err_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_lo_second);
    second.respOff.err_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_hi_second);
    
    % d) Expectation x attention: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.err_exp_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_foc);
    alldat.tg.err_exp_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_div);
    alldat.tg.err_un_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_foc);
    alldat.tg.err_un_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_div);
    alldat.tg.err_neu_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_foc);
    alldat.tg.err_neu_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_div);
    
    alldat.respOn.err_exp_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.err_exp_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.err_un_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_foc);
    alldat.respOn.err_un_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_div);
    alldat.respOn.err_neu_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_foc);
    alldat.respOn.err_neu_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_div);
    
    alldat.respOff.err_exp_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_foc);
    alldat.respOff.err_exp_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_div);
    alldat.respOff.err_un_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_foc);
    alldat.respOff.err_un_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_div);
    alldat.respOff.err_neu_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_foc);
    alldat.respOff.err_neu_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_div);
    
    % Only trials from the first half of each run
    first.tg.err_exp_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_foc_first);
    first.tg.err_exp_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_div_first);
    first.tg.err_un_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_foc_first);
    first.tg.err_un_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_div_first);
    first.tg.err_neu_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_foc_first);
    first.tg.err_neu_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_div_first);
    
    first.respOn.err_exp_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.err_exp_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.err_un_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_foc_first);
    first.respOn.err_un_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_div_first);
    first.respOn.err_neu_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_foc_first);
    first.respOn.err_neu_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_div_first);
    
    first.respOff.err_exp_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_foc_first);
    first.respOff.err_exp_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_div_first);
    first.respOff.err_un_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_foc_first);
    first.respOff.err_un_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_div_first);
    first.respOff.err_neu_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_foc_first);
    first.respOff.err_neu_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_div_first);
    
    % Only trials from the second half of each run
    second.tg.err_exp_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_foc_second);
    second.tg.err_exp_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_div_second);
    second.tg.err_un_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_foc_second);
    second.tg.err_un_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_div_second);
    second.tg.err_neu_foc{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_foc_second);
    second.tg.err_neu_div{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_div_second);
    
    second.respOn.err_exp_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.err_exp_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.err_un_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_foc_second);
    second.respOn.err_un_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_div_second);
    second.respOn.err_neu_foc{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_foc_second);
    second.respOn.err_neu_div{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_div_second);
    
    second.respOff.err_exp_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_foc_second);
    second.respOff.err_exp_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_div_second);
    second.respOff.err_un_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_foc_second);
    second.respOff.err_un_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_div_second);
    second.respOff.err_neu_foc{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_foc_second);
    second.respOff.err_neu_div{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_div_second);
    
    % e) Expectation x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.err_exp_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_lo);
    alldat.tg.err_exp_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_hi);
    alldat.tg.err_un_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_lo);
    alldat.tg.err_un_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_hi);
    alldat.tg.err_neu_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_lo);
    alldat.tg.err_neu_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_hi);
    
    alldat.respOn.err_exp_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.err_exp_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.err_un_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_lo);
    alldat.respOn.err_un_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_hi);
    alldat.respOn.err_neu_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_lo);
    alldat.respOn.err_neu_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_hi);
    
    alldat.respOff.err_exp_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_lo);
    alldat.respOff.err_exp_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_hi);
    alldat.respOff.err_un_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_lo);
    alldat.respOff.err_un_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_hi);
    alldat.respOff.err_neu_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_lo);
    alldat.respOff.err_neu_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_hi);
    
    % Only trials from the first half of each run
    first.tg.err_exp_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_lo_first);
    first.tg.err_exp_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_hi_first);
    first.tg.err_un_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_lo_first);
    first.tg.err_un_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_hi_first);
    first.tg.err_neu_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_lo_first);
    first.tg.err_neu_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_hi_first);
    
    first.respOn.err_exp_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.err_exp_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.err_un_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_lo_first);
    first.respOn.err_un_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_hi_first);
    first.respOn.err_neu_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_lo_first);
    first.respOn.err_neu_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_hi_first);
    
    first.respOff.err_exp_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_lo_first);
    first.respOff.err_exp_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_hi_first);
    first.respOff.err_un_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_lo_first);
    first.respOff.err_un_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_hi_first);
    first.respOff.err_neu_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_lo_first);
    first.respOff.err_neu_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_hi_first);
    
    % Only trials from the second half of each run
    second.tg.err_exp_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_lo_second);
    second.tg.err_exp_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_exp_hi_second);
    second.tg.err_un_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_lo_second);
    second.tg.err_un_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_un_hi_second);
    second.tg.err_neu_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_lo_second);
    second.tg.err_neu_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_neu_hi_second);
    
    second.respOn.err_exp_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.err_exp_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.err_un_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_lo_second);
    second.respOn.err_un_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_un_hi_second);
    second.respOn.err_neu_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_lo_second);
    second.respOn.err_neu_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_neu_hi_second);
    
    second.respOff.err_exp_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_lo_second);
    second.respOff.err_exp_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_exp_hi_second);
    second.respOff.err_un_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_lo_second);
    second.respOff.err_un_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_un_hi_second);
    second.respOff.err_neu_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_lo_second);
    second.respOff.err_neu_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_neu_hi_second);
    
    % f) Attention x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.err_foc_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_lo);
    alldat.tg.err_foc_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_hi);
    alldat.tg.err_div_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_lo);
    alldat.tg.err_div_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_hi);
    
    alldat.respOn.err_foc_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.err_foc_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.err_div_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_lo);
    alldat.respOn.err_div_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_hi);
    
    alldat.respOff.err_foc_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_lo);
    alldat.respOff.err_foc_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_hi);
    alldat.respOff.err_div_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_lo);
    alldat.respOff.err_div_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_hi);
    
    % Only trials from the first half of each rdiv
    first.tg.err_foc_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_lo_first);
    first.tg.err_foc_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_hi_first);
    first.tg.err_div_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_lo_first);
    first.tg.err_div_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_hi_first);
    
    first.respOn.err_foc_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.err_foc_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.err_div_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_lo_first);
    first.respOn.err_div_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_hi_first);
    
    first.respOff.err_foc_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_lo_first);
    first.respOff.err_foc_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_hi_first);
    first.respOff.err_div_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_lo_first);
    first.respOff.err_div_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_hi_first);
    
    % Only trials from the second half of each rdiv
    second.tg.err_foc_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_lo_second);
    second.tg.err_foc_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_foc_hi_second);
    second.tg.err_div_lo{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_lo_second);
    second.tg.err_div_hi{scnt} = pcat.anDiffTgLocked(:, pcat.tr_div_hi_second);
    
    second.respOn.err_foc_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.err_foc_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.err_div_lo{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_lo_second);
    second.respOn.err_div_hi{scnt} = pcat.anDiffRespLocked(:, pcat.tr_div_hi_second);
    
    second.respOff.err_foc_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_lo_second);
    second.respOff.err_foc_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_foc_hi_second);
    second.respOff.err_div_lo{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_lo_second);
    second.respOff.err_div_hi{scnt} = pcat.anDiffOffLocked(:, pcat.tr_div_hi_second);
    
    %----------------------------------------------------------------------
    % 3) Actual subjects' response on each frame
    %----------------------------------------------------------------------
    % All trials in each run
    alldat.tg.resp{scnt} = pcat.anRespTgLocked(:, tt);
    alldat.respOn.resp{scnt} = pcat.anRespRespLocked(:, tt);
    alldat.respOff.resp{scnt} = pcat.anRespOffLocked(:, tt);
    
    % Only trials from the first half of each run
    first.tg.resp{scnt} = pcat.anRespTgLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOn.resp{scnt} = pcat.anRespRespLocked(:, find(pcat.trlabel_1st2nd == 1));
    first.respOff.resp{scnt} = pcat.anRespOffLocked(:, find(pcat.trlabel_1st2nd == 1));
    
    % Only trials from the second half of each run
    second.tg.resp{scnt} = pcat.anRespTgLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOn.resp{scnt} = pcat.anRespRespLocked(:, find(pcat.trlabel_1st2nd == 2));
    second.respOff.resp{scnt} = pcat.anRespOffLocked(:, find(pcat.trlabel_1st2nd == 2));
    
    % a) Expectation: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials from each run
    alldat.tg.resp_exp{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp);
    alldat.tg.resp_un{scnt} = pcat.anRespTgLocked(:, pcat.tr_un);
    alldat.tg.resp_neu{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu);
    
    alldat.respOn.resp_exp{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp);
    alldat.respOn.resp_un{scnt} = pcat.anRespRespLocked(:, pcat.tr_un);
    alldat.respOn.resp_neu{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu);
    
    alldat.respOff.resp_exp{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp);
    alldat.respOff.resp_un{scnt} = pcat.anRespOffLocked(:, pcat.tr_un);
    alldat.respOff.resp_neu{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu);
    
    % Only trials from the first half of each run
    first.tg.resp_exp{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_first);
    first.tg.resp_un{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_first);
    first.tg.resp_neu{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_first);
    
    first.respOn.resp_exp{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_first);
    first.respOn.resp_un{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_first);
    first.respOn.resp_neu{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_first);
    
    first.respOff.resp_exp{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_first);
    first.respOff.resp_un{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_first);
    first.respOff.resp_neu{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_first);
    
    % Only trials from the second half of each run
    second.tg.resp_exp{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_second);
    second.tg.resp_un{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_second);
    second.tg.resp_neu{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_second);
    
    second.respOn.resp_exp{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_second);
    second.respOn.resp_un{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_second);
    second.respOn.resp_neu{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_second);
    
    second.respOff.resp_exp{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_second);
    second.respOff.resp_un{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_second);
    second.respOff.resp_neu{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_second);
    
    % b) Attention: tg-locked, resp onset-locked, peaked resp-locked----------
    % All trials from each run
    alldat.tg.resp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc);
    alldat.tg.resp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_div);
    
    alldat.respOn.resp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc);
    alldat.respOn.resp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_div);
    
    alldat.respOff.resp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc);
    alldat.respOff.resp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_div);
    
    % Only trials from the first half of each run
    first.tg.resp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_first);
    first.tg.resp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_first);
    
    first.respOn.resp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_first);
    first.respOn.resp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_first);
    
    first.respOff.resp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_first);
    first.respOff.resp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_first);
    
    % Only trials from the second half of each run
    second.tg.resp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_second);
    second.tg.resp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_second);
    
    second.respOn.resp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_second);
    second.respOn.resp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_second);
    
    second.respOff.resp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_second);
    second.respOff.resp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_second);
    
    % c) Coherence: tg-locked, resp onset-locked, peaked resp-locked----------
    % All trials from each run
    alldat.tg.resp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_lo);
    alldat.tg.resp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_hi);
    
    alldat.respOn.resp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_lo);
    alldat.respOn.resp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_hi);
    
    alldat.respOff.resp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_lo);
    alldat.respOff.resp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_hi);
    
    % Only trials from the first half of each run
    first.tg.resp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_lo_first);
    first.tg.resp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_hi_first);
    
    first.respOn.resp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_lo_first);
    first.respOn.resp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_hi_first);
    
    first.respOff.resp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_lo_first);
    first.respOff.resp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_hi_first);
    
    % Only trials from the second half of each run
    second.tg.resp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_lo_second);
    second.tg.resp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_hi_second);
    
    second.respOn.resp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_lo_second);
    second.respOn.resp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_hi_second);
    
    second.respOff.resp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_lo_second);
    second.respOff.resp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_hi_second);
    
    % d) Expectation x attention: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.resp_exp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_foc);
    alldat.tg.resp_exp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_div);
    alldat.tg.resp_un_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_foc);
    alldat.tg.resp_un_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_div);
    alldat.tg.resp_neu_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_foc);
    alldat.tg.resp_neu_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_div);
    
    alldat.respOn.resp_exp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.resp_exp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc);
    alldat.respOn.resp_un_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_foc);
    alldat.respOn.resp_un_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_div);
    alldat.respOn.resp_neu_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_foc);
    alldat.respOn.resp_neu_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_div);
    
    alldat.respOff.resp_exp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_foc);
    alldat.respOff.resp_exp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_div);
    alldat.respOff.resp_un_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_foc);
    alldat.respOff.resp_un_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_div);
    alldat.respOff.resp_neu_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_foc);
    alldat.respOff.resp_neu_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_div);
    
    % Only trials from the first half of each run
    first.tg.resp_exp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_foc_first);
    first.tg.resp_exp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_div_first);
    first.tg.resp_un_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_foc_first);
    first.tg.resp_un_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_div_first);
    first.tg.resp_neu_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_foc_first);
    first.tg.resp_neu_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_div_first);
    
    first.respOn.resp_exp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.resp_exp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc_first);
    first.respOn.resp_un_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_foc_first);
    first.respOn.resp_un_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_div_first);
    first.respOn.resp_neu_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_foc_first);
    first.respOn.resp_neu_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_div_first);
    
    first.respOff.resp_exp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_foc_first);
    first.respOff.resp_exp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_div_first);
    first.respOff.resp_un_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_foc_first);
    first.respOff.resp_un_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_div_first);
    first.respOff.resp_neu_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_foc_first);
    first.respOff.resp_neu_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_div_first);
    
    % Only trials from the second half of each run
    second.tg.resp_exp_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_foc_second);
    second.tg.resp_exp_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_div_second);
    second.tg.resp_un_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_foc_second);
    second.tg.resp_un_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_div_second);
    second.tg.resp_neu_foc{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_foc_second);
    second.tg.resp_neu_div{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_div_second);
    
    second.respOn.resp_exp_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.resp_exp_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_foc_second);
    second.respOn.resp_un_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_foc_second);
    second.respOn.resp_un_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_div_second);
    second.respOn.resp_neu_foc{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_foc_second);
    second.respOn.resp_neu_div{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_div_second);
    
    second.respOff.resp_exp_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_foc_second);
    second.respOff.resp_exp_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_div_second);
    second.respOff.resp_un_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_foc_second);
    second.respOff.resp_un_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_div_second);
    second.respOff.resp_neu_foc{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_foc_second);
    second.respOff.resp_neu_div{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_div_second);
    
    % e) Expectation x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.resp_exp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_lo);
    alldat.tg.resp_exp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_hi);
    alldat.tg.resp_un_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_lo);
    alldat.tg.resp_un_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_hi);
    alldat.tg.resp_neu_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_lo);
    alldat.tg.resp_neu_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_hi);
    
    alldat.respOn.resp_exp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.resp_exp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo);
    alldat.respOn.resp_un_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_lo);
    alldat.respOn.resp_un_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_hi);
    alldat.respOn.resp_neu_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_lo);
    alldat.respOn.resp_neu_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_hi);
    
    alldat.respOff.resp_exp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_lo);
    alldat.respOff.resp_exp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_hi);
    alldat.respOff.resp_un_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_lo);
    alldat.respOff.resp_un_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_hi);
    alldat.respOff.resp_neu_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_lo);
    alldat.respOff.resp_neu_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_hi);
    
    % Only trials from the first half of each run
    first.tg.resp_exp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_lo_first);
    first.tg.resp_exp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_hi_first);
    first.tg.resp_un_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_lo_first);
    first.tg.resp_un_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_hi_first);
    first.tg.resp_neu_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_lo_first);
    first.tg.resp_neu_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_hi_first);
    
    first.respOn.resp_exp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.resp_exp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo_first);
    first.respOn.resp_un_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_lo_first);
    first.respOn.resp_un_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_hi_first);
    first.respOn.resp_neu_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_lo_first);
    first.respOn.resp_neu_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_hi_first);
    
    first.respOff.resp_exp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_lo_first);
    first.respOff.resp_exp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_hi_first);
    first.respOff.resp_un_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_lo_first);
    first.respOff.resp_un_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_hi_first);
    first.respOff.resp_neu_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_lo_first);
    first.respOff.resp_neu_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_hi_first);
    
    % Only trials from the second half of each run
    second.tg.resp_exp_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_lo_second);
    second.tg.resp_exp_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_exp_hi_second);
    second.tg.resp_un_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_lo_second);
    second.tg.resp_un_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_un_hi_second);
    second.tg.resp_neu_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_lo_second);
    second.tg.resp_neu_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_neu_hi_second);
    
    second.respOn.resp_exp_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.resp_exp_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_exp_lo_second);
    second.respOn.resp_un_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_lo_second);
    second.respOn.resp_un_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_un_hi_second);
    second.respOn.resp_neu_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_lo_second);
    second.respOn.resp_neu_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_neu_hi_second);
    
    second.respOff.resp_exp_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_lo_second);
    second.respOff.resp_exp_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_exp_hi_second);
    second.respOff.resp_un_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_lo_second);
    second.respOff.resp_un_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_un_hi_second);
    second.respOff.resp_neu_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_lo_second);
    second.respOff.resp_neu_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_neu_hi_second);
    
    % f) Attention x coherence: tg-locked, resp onset-locked, peaked resp-locked--------
    % All trials in each run
    alldat.tg.resp_foc_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_lo);
    alldat.tg.resp_foc_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_hi);
    alldat.tg.resp_div_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_lo);
    alldat.tg.resp_div_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_hi);
    
    alldat.respOn.resp_foc_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.resp_foc_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo);
    alldat.respOn.resp_div_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_lo);
    alldat.respOn.resp_div_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_hi);
    
    alldat.respOff.resp_foc_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_lo);
    alldat.respOff.resp_foc_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_hi);
    alldat.respOff.resp_div_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_lo);
    alldat.respOff.resp_div_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_hi);
    
    % Only trials from the first half of each rdiv
    first.tg.resp_foc_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_lo_first);
    first.tg.resp_foc_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_hi_first);
    first.tg.resp_div_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_lo_first);
    first.tg.resp_div_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_hi_first);
    
    first.respOn.resp_foc_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.resp_foc_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo_first);
    first.respOn.resp_div_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_lo_first);
    first.respOn.resp_div_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_hi_first);
    
    first.respOff.resp_foc_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_lo_first);
    first.respOff.resp_foc_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_hi_first);
    first.respOff.resp_div_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_lo_first);
    first.respOff.resp_div_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_hi_first);
    
    % Only trials from the second half of each rdiv
    second.tg.resp_foc_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_lo_second);
    second.tg.resp_foc_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_foc_hi_second);
    second.tg.resp_div_lo{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_lo_second);
    second.tg.resp_div_hi{scnt} = pcat.anRespTgLocked(:, pcat.tr_div_hi_second);
    
    second.respOn.resp_foc_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.resp_foc_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_foc_lo_second);
    second.respOn.resp_div_lo{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_lo_second);
    second.respOn.resp_div_hi{scnt} = pcat.anRespRespLocked(:, pcat.tr_div_hi_second);
    
    second.respOff.resp_foc_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_lo_second);
    second.respOff.resp_foc_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_foc_hi_second);
    second.respOff.resp_div_lo{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_lo_second);
    second.respOff.resp_div_hi{scnt} = pcat.anRespOffLocked(:, pcat.tr_div_hi_second);
    
    cd ../..
end

% %plotting all responses just to see
% for ti = 1:size(pcat.distance0, 2)
%     figure(ceil(ti/120));
%     if mod(ti, 120) == 0 %last subplot (120th) of each fig
%         subplot(6, 20, 120); plot(pcat.distance0(:, ti), 'b'); title(['tr ', num2str(ti)]); hold on;
%         line([pcat.resp_index(ti), pcat.resp_index(ti)], [0, 1], 'Color', 'red'); %drawing a line to indicate resp onset
%         ylim([0 1]); drawnow;
%     else
%         subplot(6, 20, mod(ti, 120)); plot(pcat.distance0(:, ti), 'b'); title(['tr ', num2str(ti)]); hold on;
%         line([pcat.resp_index(ti), pcat.resp_index(ti)], [0, 1], 'Color', 'red'); %drawing a line to indicate resp onset
%         ylim([0 1]); drawnow;
%     end
% end

%--------------------------------------------------------------------------
%% Save out stacked data from all subjects
%--------------------------------------------------------------------------
% This is the manuscript version: behavDat_050620.mat
%save('behavDat_050620.mat', 'data', 'numfr', 'allsub', 'all', 'pcat', ...
%    'tgOnset_before', 'tgOnset_after', 'respOnset_before', 'respOnset_after', ...
%    'respOffset_before', 'respOffset_after');

% This is the same as above but but just a bit more organized and has 4
% versions of what trials we want to keep. For the behavDat_050630.mat
% version, early and missed trials were discarded. Here, depending on
% 'ana', we either do that or still keep early trials. Same with 'opp' and
% trials with responses in the opposite directions
save(outFile, 'numfr', 'allsub', 'alldat', 'first', ...
    'second', 'pcat', 'tgOnset_before', 'tgOnset_after', 'respOnset_before', ...
    'respOnset_after', 'respOffset_before', 'respOffset_after', 'ana', 'opp');
