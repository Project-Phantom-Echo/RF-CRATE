for i = 58:61
    try
        filename = sprintf('wificsi1/wificsi1_exp%03d.mat', i);
        disp(filename);
        wifi_csi = load(filename).wifi_csi_1;
        ann = table2array(wifi_csi(:,1:end-270));
        csi= single(table2array(wifi_csi(:,end-269:end)));
        header = wifi_csi.Properties.VariableNames;

        save(sprintf('/media/zx/ssd/wificsi1-py/wificsi1_exp%03d.mat', i), 'ann', 'csi', 'header', '-v7.3'); 
    catch
        fprintf('Error in %d\n', i);
    end
end


for i = 1:63
    try
        filename = sprintf('wificsi2/wificsi2_exp%03d.mat', i);
        disp(filename);
        wifi_csi = load(filename).wifi_csi_2;
        ann = table2array(wifi_csi(:,1:end-270));
        csi= single(table2array(wifi_csi(:,end-269:end)));
        header = wifi_csi.Properties.VariableNames;
        save(sprintf('/media/zx/ssd/wificsi2-py/wificsi2_exp%03d.mat', i), 'ann', 'csi', 'header', '-v7.3'); 
    catch
        fprintf('Error in %d\n', i);
    end
end

% conver strings to char

for i = 1:61
    try
        filename = sprintf('/media/zx/ssd/wificsi1-py/wificsi1_exp%03d.mat', i);
        disp(filename);
        data = load(filename);
        ann = convertStringsToChars(data.ann);
        % csi= data.csi;
        header = data.header;
        save(sprintf('/media/zx/ssd/wificsi1-py-ann/wificsi1_exp%03d.mat', i), 'ann', 'header', '-v7.3'); 
    catch
        fprintf('Error in %d\n', i);
    end
end


for i = 1:63
    try
        filename = sprintf('/media/zx/ssd/wificsi2-py/wificsi2_exp%03d.mat', i);
        disp(filename);
        data = load(filename);
        ann = convertStringsToChars(data.ann);
        % csi= data.csi;
        header = data.header;
        save(sprintf('/media/zx/ssd/wificsi2-py-ann/wificsi2_exp%03d.mat', i), 'ann', 'header', '-v7.3'); 
    catch
        fprintf('Error in %d\n', i);
    end
end




for i = 1:63
    try
        filename = sprintf('wificsi2/wificsi2_exp%03d.mat', i);
        disp(filename);
        wifi_csi = load(filename).wifi_csi_2;
        ann_cell = table2cell(wifi_csi(:,1:end-270));

        csv_file = fopen(sprintf('/media/zx/ssd/wificsi2-csv/wificsi2_exp%03d.tsv', i), 'w+');

        for row = 1:size(ann_cell,1)
            ann_str = "";
            for col = 1:size(ann_cell,2)
                ann_val = ann_cell{row, col};
                % if it is <missing> then replace it with a NaN
                if ismissing(ann_val)
                    ann_val = "NaN";
                end
                ann_str = strcat(ann_str, sprintf('%s|', ann_val));
            end
            ann_str = strcat(ann_str, "\n");
            fprintf(csv_file, ann_str);
        end
    % catch error and print stack trace
    % <missing> string element not supported.
    % if 
    catch ME
        fprintf('Error in %d\n', i)
        disp(getReport(ME));
    end
end

for i = 1:61
    try
        filename = sprintf('wificsi1/wificsi1_exp%03d.mat', i);
        disp(filename);
        wifi_csi = load(filename).wifi_csi_1;
        ann_cell = table2cell(wifi_csi(:,1:end-270));

        csv_file = fopen(sprintf('/media/zx/ssd/wificsi1-csv/wificsi1_exp%03d.tsv', i), 'w+');

        for row = 1:size(ann_cell,1)
            ann_str = "";
            for col = 1:size(ann_cell,2)
                ann_val = ann_cell{row, col};
                % if it is <missing> then replace it with a NaN
                if ismissing(ann_val)
                    ann_val = "NaN";
                end
                ann_str = strcat(ann_str, sprintf('%s|', ann_val));
            end
            ann_str = strcat(ann_str, "\n");
            fprintf(csv_file, ann_str);
        end

    catch ME
        fprintf('Error in %d\n', i)
        disp(getReport(ME));
    end
end


i = 61;
filename = sprintf('wificsi1/wificsi1_exp%03d.mat', i);
disp(filename);
wifi_csi = load(filename).wifi_csi_1;
ann_cell = table2cell(wifi_csi(:,1:end-270));

csv_file = fopen(sprintf('/media/zx/ssd/wificsi1-csv/wificsi1_exp%03d.tsv', i), 'w+');

for row = 2181837:size(ann_cell,1)
    ann_str = "";
    for col = 1:size(ann_cell,2)
        ann_str = strcat(ann_str, sprintf('%s|', ann_cell{row, col}));
    end
    ann_str = strcat(ann_str, "\n");
    fprintf(csv_file, ann_str);
end