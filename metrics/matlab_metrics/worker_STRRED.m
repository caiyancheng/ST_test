function worker_STRRED()

fprintf( 1, '<start>\n' );

%fprintf( 2, '%s\n', pwd );

if ~exist( 'run_strred', 'file' )
    strred_path = fullfile( pwd, 'strred' );
    if ~isdir( strred_path )
        fprintf( 2, "Missing STRRED directory: %s", strred_path );
        return;
    end
    addpath( genpath(strred_path) );
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
        %fprintf( 2, 'Reading "%s"\n', mat_file );

        frames = load( mat_file );
        %fprintf( 2, '%d, ', size(frames.Yr) )

        if strcmp(frames.type, 'image')
            [spatial_ref, temporal_ref] = extract_info( gpuArray(frames.Yr), gpuArray(frames.Yr_prev) );
            [spatial_dis, temporal_dis] = extract_info( gpuArray(frames.Yt), gpuArray(frames.Yt_prev) );
            srred = mean2(abs(spatial_ref - spatial_dis));
            trred = mean2(abs(temporal_ref - temporal_dis));
        else
        
            [spatial_ref, temporal_ref] = extract_info( gpuArray(frames.Yr), gpuArray(frames.Yr_prev) );
            [spatial_dis, temporal_dis] = extract_info( gpuArray(frames.Yt), gpuArray(frames.Yt_prev) );
            srred = mean2(abs(spatial_ref - spatial_dis));
            trred = mean2(abs(temporal_ref - temporal_dis));

        end 

        fprintf( 1, '%g %g \n', srred, trred );
        %fprintf( 1, '%g\n', srred );
    else
        error( 'Unknown command "%s"', cmd );
    end
end

