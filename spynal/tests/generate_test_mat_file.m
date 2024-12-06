function generate_test_mat_file(basename, versions)
  % generate_test_mat_file  Generates mat file(s) with several basic data types for testing
  %
  % Used for generating mat files for unit tests of Python matIO module
  %
  % ARGS
  % basename  String. Full-path base basename of file(s) to generate.  
  %           Should not contain extension. Version string and .mat ext will be appended.
  %           Default: './testing_datafile_' (eg generating './testing_datafile_v73.mat')
  %
  % versions  Cellstr. List of mat file versions to create: 'v7' | 'v73'
  %           Default: {'v7','v73'}

  if (nargin < 1) || isempty(basename), basename = './testing_datafile_'; end 
  if (nargin < 2) || isempty(versions), versions = {'v7','v73'}; end 

  if ischar(versions), versions = {versions}; end
    
  % Scalar integer, float, and string types
  integer     = int16(1);
  floating    = double(1.1);  % Note: have to add decimal value else scipy.io loads as int
  boolean     = true;
  string      = 'abc';

  % Numerical array [4 x 3 x 2]. Note: have to add decimal value else scipy.io loads as int
  num_array   = reshape(1:6, [1,2,3]) + 0.1;
  % DEL num_array   = reshape(1:24, [4,3,2]) + 0.1;

  % TODO Add datatype of same-width char array
  % % cell array of strings [4x6]
  % cell_array  = reshape(num2cell('a':'x'), [4,3,2]);

  % cell array of strings [4x6]
  cell_array  = repmat({'abc','def','gh','ij'}',[1,6]); % TODO Change to [4,3,2]

  % Generic struct with fields of all different types
  gen_struct  = struct('integer',integer, 'floating',floating, 'boolean',boolean, 'string',string, ...
                       'num_array',num_array, 'cell_array',{cell_array});

  % Table-like struct w/ [4 x 1] numerical and cellstr fields
  table_struct= struct('num_array',num_array(1:4)', 'cell_array',{cell_array(:,1,1)});
  % DEL table_struct= struct('num_array',num_array(:,1,1), 'cell_array',{cell_array(:,1,1)});

  variables = {'integer','floating','boolean','string', 'num_array','cell_array', 'gen_struct','table_struct'};

  % Save mat file of each requested version
  for i_version = 1:length(versions); version = versions{i_version};
    if strcmp(version,'v7'),      version_tag = '-v7';
    elseif strcmp(version,'v73'), version_tag = '-v7.3';
    else
      error('Unknown mat file version ''%s''. Should be ''v7''|''v73''', version);
    end

    filename = [basename, version, '.mat'];
    save(filename, version_tag, variables{:});
  end
end