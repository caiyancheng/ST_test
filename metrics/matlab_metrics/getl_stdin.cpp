#include <matrix.h>  /* Matlab matrices */
#include <mex.h>

#include <iostream>
#include <string>

#include <string.h>

void mexFunction(int nlhs,	     /* Num return vals on lhs */
		 mxArray *plhs[],    /* Matrices on lhs      */
		 int nrhs,	     /* Num args on rhs    */
		 const mxArray *prhs[]     /* Matrices on rhs */
		 )
  {

    std::string line;
    std::getline(std::cin, line);
 
    char *output_buf = (char*)mxCalloc(line.length()+1, sizeof(char));
    strcpy( output_buf, line.c_str() );

    if( nlhs == 1 )
      plhs[0] = mxCreateString(output_buf);

  }
