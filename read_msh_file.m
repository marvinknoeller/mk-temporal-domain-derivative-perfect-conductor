function [nodes, elements] = read_msh_file(filename)
    % Open the file
    fid = fopen(filename, 'r');
    
    if fid == -1
        error('Cannot open the file.');
    end
    
    % Initialize variables
    nodes = [];
    elements = [];
    
    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);
        
        if contains(line, '$Nodes')
            % Read the number of nodes
            num_nodes = str2double(fgetl(fid));
            nodes = zeros(num_nodes, 3);
            
            % Read node coordinates
            for i = 1:num_nodes
                node_line = fgetl(fid);
                node_data = sscanf(node_line, '%d %f %f %f');
                nodes(i, :) = node_data(2:4)';
            end
        elseif contains(line, '$Elements')
            % Read the number of elements
            num_elements = str2double(fgetl(fid));
            elements = [];
            
            % Read elements
            for i = 1:num_elements
                elem_line = fgetl(fid);
                elem_data = sscanf(elem_line, '%d %d %d %d %d %d');
                
                % Check for triangles (element type 2)
                if elem_data(2) == 2
                    elements = [elements; elem_data(end-2:end)'];
                end
            end
        end
    end
    
    % Close the file
    fclose(fid);
end
