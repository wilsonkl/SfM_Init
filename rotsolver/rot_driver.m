function rot_driver(EGs_file, rot_soln, cc_file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION rot_driver
%
% Run Chatterjee and Govindu's L1RA pairwise rotations solver with typical bundler-style
% files. 
%
% INPUTS:
%   EGs_file:         input graph, where each edge is an epipolar geometry
%   rot_soln:         file to write output solution to
%   cc_file:          a list of vertices to use. The EGs graph will be 
%                         restricted to this subset of vertices before computing
% 
% FILE FORMATS:
%   Bundler style rotation matrices are world-to-camera maps. In bundler terminology, the
%   transpose of a camera's rotation matrix is that camera's pose. Bundler pairwise 
%   rotations are defined as the pose of camera j in camera i's coordinate system. That
%   is, Rij = Ri * Rj'. 
%
%   EGs files:
%       epipolar geometries are listed one pair per line:
%       i j Rij [tij]
%       Rij is a pairwise rotation as defined above printed row-major, and tij is a unit 
%       3-vector. Note that for the purposes of this program, tij is optional.
%   rot files:
%       rotation matrices, one per line:
%       i Rij
%   cc file: 
%       integers, one per line
%
% 2014 Kyle Wilson, Cornell University
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read the input graph
fprintf('[rot_driver] reading EGs from %s\n', EGs_file)
[edges, pairwise_rots, ~] = read_EGs_file(EGs_file);
edges = edges + 1; % zero-indexing -> 1-indexing
fprintf('[rot_driver] found a graph with %d nodes and %d edges\n', ...
    length(unique(edges)), size(edges, 2));

% check that the rotations really are rotations
for i=1:size(pairwise_rots,3)
    R = pairwise_rots(:,:,i);
    if abs(norm(R'*R)-1) > 1e-3
        fprintf('[rot_driver] bad rotation');
    end
end

% read the cc file
fid = fopen(cc_file, 'r');
cc = textscan(fid, '%d');
cc = cc{1} + 1; % zero-indexing -> 1-indexing
fclose(fid);
fprintf('[rot_driver] read a cc file with %d nodes\n', length(cc));

% restrict the graph
ind = ismember(edges(1,:), cc) & ismember(edges(2,:), cc);
edges = edges(:,ind);
pairwise_rots = pairwise_rots(:,:,ind);
fprintf('[rot_driver] after applying cc, there are %d nodes and %d edges\n', ...
    length(unique(edges)), size(edges, 2));

% transpose each pairwise rotation?
pairwise_rots = permute(pairwise_rots, [2 1 3]);
 
% do an internal connected component calculation
ccs = find_ccs(edges, 2);

% only crunch on the biggest
cc_sizes = zeros(length(ccs),1);
for i=1:length(ccs)
    cc_sizes(i) = length(ccs{i});
end
[~,imax] = max(cc_sizes);
cc_internal = ccs{imax};

% restrict the graph to the largest internal cc
ind = ismember(edges(1,:), cc_internal);
edges_cc = edges(:,ind);
pairwise_rots_cc = pairwise_rots(:,:,ind);
fprintf('[rot_driver] after applying another internal cc, there are %d nodes and %d edges\n', ...
    length(unique(edges_cc)), size(edges_cc, 2));

% the solver code doesn't like unused vertices. We need to 
% rename each vertex so that 1:n are all used for some n.
[edges_cc, invkey] = reindex_graph(edges_cc);

% solve the problem
Rb = BoxMedianSO3Graph(pairwise_rots_cc, edges_cc);
Rm = RobustMeanSO3Graph(pairwise_rots_cc, edges_cc, 5, Rb);

% write answer to output, switch back to zero-indexing
write_rot_file(rot_soln, Rm, invkey-1);    

end % rot_driver

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%


function [edges_new, invkey] = reindex_graph(edges)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION reindex_graph
%
% take a graph where vertices are numbered as a subset of 
% 1..n1 and return a new graph where vertices are exactly
% the set 1..n2, where n2 <= n1.
%
% INPUTS:
%   edges: input graph, given as a 2-by-n matrix, where
%       columns are edges [i;j]
%
% RETURNS:
%   edges_new: new edges, with the substitution of key(i)
%       for every occurance of i
%   invkey: a map from edges_new to edges, i.e. 
%       edges(invkey(i),:) == edges_new(i,:)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vertices_old = unique(edges)';
vertices_new = int32(1:length(vertices_old));
key = repmat(-1,1,max(vertices_old));
key(vertices_old) = vertices_new;

edges_new = [key(edges(1,:)); key(edges(2,:))];
invkey = vertices_old;
end % reindex_graph

function ccs = find_ccs(edges, min_cc_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION find_ccs
% 
% return all connected components in the undirected graph
% given by an edge list.
%
% INPUTS:
%   edges: n-by-2 matrix of non-negative integers
%   min_cc_size: discard all ccs smaller than this
%
% RETURNS:
%   ccs: num_ccs-by-1 cell array. Each ccs{i} is a list of
%       vertices in conected component i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_v = max(max(edges));
num_e = size(edges, 12);
i = double(edges(1,:));
j = double(edges(2,:));
G = sparse([i;j], [j;i], 1.0);
% this function is from the bioinformatics toolbox...?
[num_ccs, labels] = graphconncomp(G, 'Directed', false);

ccs = cell(num_ccs,1);
for i = 1:length(labels)
    ccs{labels(i)} = [ccs{labels(i)} i];
end

ccs_new = {};
for i = 1:num_ccs
    if length(ccs{i}) >= min_cc_size
        ccs_new{end+1} = ccs{i};
    end
end
ccs = ccs_new;
end % find_ccs 

function [edges, rots, trans] = read_EGs_file(fname)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION read_EGs_file
%
% return the columns in fname, according to the file format
% defined above. 
% 
% RETURNS:
%   edges: nlines-by-2 matrix, rows are edge indices [i j]
%   rots:  nlines*3-by-3 matrix: stacked 3-by-3 blocks
%   trans: nlines-by-3 matrix, rows are unit vectors tij
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread(fname);
[nlines, nfields] = size(data);

edges = int32(data(:,1:2)).';

rots = zeros(3,3,nlines);
for i = 1:nlines
    rots(:,:,i) = reshape(data(i,3:11),3,3).';
end

if nfields < 14
    trans = [];
else
    trans = data(:,12:14).';
end
end % read_EGs_file

function write_rot_file(fname, rots, indices)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION write_rot_file
%
% write a rotations file in the format described above.
%
% INPUTS:
%   fname: file to write
%   rots: 3n-by-3 matrix, where each 3-by-3 block is a 
%       rotation matrix
%   indices: length n column vector associating an id number
%       to each rotation matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[fid, errmsg] = fopen(fname, 'w');
if fid == -1
    disp(errmsg)
    error(['File could not be opened for write: ' fname])
end
n = length(indices);
if size(rots,3) ~= n
    error('write_rot_file: input sizes do not match')
end

for i = 1:n
    fprintf(fid, '%d %f %f %f %f %f %f %f %f %f\n', ...
        indices(i), rots(:,:,i).');
end
fclose(fid);
end % write_rot_file
