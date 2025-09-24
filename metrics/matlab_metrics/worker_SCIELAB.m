function worker_STRRED()

fprintf( 1, '<start>\n' );

%fprintf( 2, '%s\n', pwd );

if ~exist( 'visualAngle', 'file' )
    scielab_path = fullfile( pwd, 'scielab_functions' );
    if ~isdir( scielab_path )
        fprintf( 2, "Missing SCIELAB directory: %s", strred_path );
        return;
    end
    addpath( genpath(scielab_path) );
end

warning('off','MATLAB:singularMatrix');

while true

    %cmd = input( "", "s" );
    cmd = getl_stdin();
    
    if isempty(cmd)
        continue;
    end

    if cmd(1)=='q'
        break;
    elseif cmd(1)=='c' % compare
        C = strsplit( cmd );
        mat_file = C{2};

        frames = load( mat_file );

        ppd = double(frames.ppd);
        wp = whitepoint( 'd65' );

        T_xyz = cm_rgb2xyz(srgb2lin(frames.T), 'rec709');
        R_xyz = cm_rgb2xyz(srgb2lin(frames.R), 'rec709');

        % T_xyz = pfs_transform_colorspace( 'sRGB', frames.T, 'XYZ' );
        % R_xyz = pfs_transform_colorspace( 'sRGB', frames.R, 'XYZ' );

        
        DEmap = scielab(ppd, gpuArray(T_xyz), gpuArray(R_xyz), wp, 'xyz');

        DEmean = mean(DEmap(:));

        fprintf( 1, '%g\n', DEmean );

    else
        error( 'Unknown command "%s"', cmd );
    end

end

