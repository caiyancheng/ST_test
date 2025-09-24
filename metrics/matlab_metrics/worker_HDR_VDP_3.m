function worker_HDR_VDP_3()

fprintf( 1, '<start>\n' );

if ~exist( 'hdrvdp3', 'file' )
    hdrvdp_path = fullfile( pwd, 'hdrvdp' );
    if ~isdir( hdrvdp_path )
        fprintf( 2, "Missing HDR-VDP-3 directory: %s", hdrvdp_path );
        return;
    end
    addpath( genpath(hdrvdp_path) );
end

while true

    cmd = getl_stdin();

    if isempty(cmd)
        continue;
    end

    if cmd(1)=='q'
        break;
    elseif cmd(1)=='c' % compare

        mat_file = cmd(3:end);

        frames = load( mat_file );
        
        hdrvdp_options = { 'disable_lowvals_warning', true, 'quiet', true };
        df = hdrvdp3( 'quality', frames.T, frames.R, 'rgb-bt.2020', double(frames.ppd), hdrvdp_options );

        Q = df.Q_JOD;
        fprintf( 1, '%g\n', Q );
    else
        error( 'Unknown command "%s"', cmd );
    end
end

