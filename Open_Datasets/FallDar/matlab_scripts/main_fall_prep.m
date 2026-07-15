% file_prefix = "/mnt/data/Widar3.0ReleaseData/CSI/20181109/";
file_prefix = "FALL_CSI_DAT";
filelist = dir(file_prefix + "/*/*.dat");
% filelist = dir("/mnt/data/Widar3.0ReleaseData/MAT/20181109/user2/user2-5-3-1-12.mat");
save_prefix = "/mnt/data/Widar3.0ReleaseData/FALL_Seg/";
normal_folder = ["20200525", "20200526", "20200623", "20200706", "20200712"];
save_prefix = "FALL_MAT";
bad_ssfull_file = ["Data_SS/20200614/fall_SSfull_bionic_20200614_1_5.mat"];
% processeds = zeros(5520, 200,60);

% File pattern 
% preprocessed:
% nonfall_CSIAMP_zhangyi_20200525_22_2_6 
% <label>_CSIAMP_<name>_<date>_<activity>_<repeat>_<segment_id_by_1.5s_window>
% fall_CSIAMP_bionic_20200601_1_89
% <label>_CSIAMP_<name>_<date>_<activity>_<repeat>
% Raw data
% zhangyi-1-1-r
% <name>-<activity>-<repeat>-r


k=1;
for f_id = 1:length(filelist)
    file_struct = filelist(f_id);
    folder_parts = split(file_struct.folder,"/");
    f_date = folder_parts(7);
    if (any(normal_folder==f_date{1,1}))
        continue;
    end
    csi_data = csi_get_all(file_struct.folder + "/" + file_struct.name);
    name_parts = split(file_struct.name, "-");
    
    person_name = name_parts(1);
    idx1 = name_parts(2);
    idx2 = name_parts(3);
    
    enable_norm = 1;
    enable_filter = 1;
  
    if any(normal_folder==f_date{1,1})
        for win_start = 1:750:size(csi_data,1)-1499
            label(k)=1;
            [pp, ms_one] = preproc_matrix(csi_data(win_start:win_start+1499,:), 1000, 1, 3, 256, 10, 1:30, enable_norm, enable_filter);
            data_pp{k} = pp;
            ms(k,:) = ms_one';
            k=k+1;
        end
    else
        try
            label(k)=2;
            fall_ssfull_file = ['Data_SS/', f_date, '/fall_SSfull_', person_name, '_', f_date, '_', idx1, '_', idx2, '.mat'];
            load(convertCharsToStrings(join(fall_ssfull_file,"")));
            [max_v, max_i] = max(speed_sequence);
            start_ts = max_i - 999;
            end_ts = max_i + 500;
            if ~(start_ts >= 1 && end_ts <= size(csi_data,1))
                end_ts - start_ts;
                disp(convertCharsToStrings(join(fall_ssfull_file,"")));
                continue;
            end
            [pp, ms_one] = preproc_matrix(csi_data(start_ts:end_ts,:), 1000, 1, 3, 256, 10, 1:30, enable_norm, enable_filter);
            data_pp{k} = pp;
            ms(k,:) = ms_one';
        catch
            disp(convertCharsToStrings(join(fall_ssfull_file,"")));
            continue
        end
        k=k+1;
    end
    disp(k)
end
% save("./processeds", "processeds");
% save("./mss", "mss");

function mysave_processed(filepath, data, ms)
    save(filepath, "data", "ms");
end