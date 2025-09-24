function worker_SPEEDQA()

fprintf( 1, '<start>\n' );

%fprintf( 2, '%s\n', pwd );

if ~exist( 'Single_Scale_SpEED', 'file' )
    speedqa_path = fullfile( pwd, 'speedqa' );
    if ~isdir( speedqa_path )
        fprintf( 2, "Missing SPEED-QA directory: %s", speedqa_path );
        return;
    end
    addpath( genpath(speedqa_path) );
end

warning('off','MATLAB:singularMatrix');
warning('off','all');

%%%% SpEED parameters
sigma_nsq = 0.1;
window = fspecial('gaussian', 7, 7/6);
window = window/sum(sum(window));

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

            blk_speed = 3;
            down_size_ss = 2;

            [SPEED_ss, SPEED_ss_SN] = Single_Scale_SpEED(double(frames.Yr), double(frames.Yt), down_size_ss, blk_speed, window, sigma_nsq);

            fprintf( 1, '%g %g \n', SPEED_ss, 1.0 );

        else

            blk_speed = 5;
            down_size = 4;

            [speed_s, speed_s_sn, speed_t, speed_t_sn] = Single_Scale_Video_SPEED(double(frames.Yr_prev), double(frames.Yr), double(frames.Yt_prev), double(frames.Yt), down_size, window, blk_speed, sigma_nsq);

            fprintf( 1, '%g %g \n', speed_s, speed_t );
        end
    else
        error( 'Unknown command "%s"', cmd );
    end
end

