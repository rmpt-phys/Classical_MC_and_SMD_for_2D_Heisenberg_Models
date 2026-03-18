	      /*------------------------------------
	Prepare binary file for the gap-size
	data measurements and statistics ... */
     
      string pointTag[4] = { "Kp" , "Mx" , "My", "Gp" };

      string specType[4] = { "xy" , "yz" , "zz", "tt" };
      
      string col, varName, fname = outDir1 + subDir2;

      ofstream gapSzFile;

      if (iAmRoot)
	{
	  fname += "GapSzData" + tTagDat;

	  gapSzFile.open(fname);

	  for (i = 0; i < 4; i++)
	    {
	      n0 = 4 * i;
	      
	      for (j = 0; j < 4; j++)
		{
		  n = n0 + (j + 1);
		  
		  col = iformat2(n);
		  
		  varName = "#  " + pointTag[i] + " SPC:";

		  varName += specType[j] + " c" + col;
	      		  
		  gapSzFile << varName << X2;
		}
	    }///| Record data information;

	  gapSzFile << endl;
	}
	
	    //XXXXXXXXXX
	    //XXXXXXXXXX
	    	
	  /*.......................................
	    Perform measurement of the dynamical SF

	    Notes about the procedure:
    
	    1) The current spin configuration is used as
	    input for the real time evolution of the 2D
	    lattice system, which is described by the
	    dynamics of the associated Heisenberg
	    equations (coupled ODEs);

	    2) The procedure below is based on the 4th
	    order RK method, the resulting data within
	    the 'tSeries' type pointers is then accumu-
	    lated in the root-only CMT-type arrays;

	    3) The gap-size vectors below are only rele-
	    vant when the model is not SO(3) symmetric:
	    parameter 'lambda' is finite ( XXZ model ); */

	  double gapSzVec[16];
	  double vecSum4d[16];
	  
	  get_dynSpectrum(wRank, tmWinVec, spinField,
			  SqxWVec, SqyWVec, SqzWVec,
			  qSStVec, gapSzVec);

	  MPI_Reduce(gapSzVec, vecSum4d, 16, MPI_DOUBLE,
		     MPI_SUM, root, MPI_COMM_WORLD);
       
	  if (iAmRoot)
	    {
	      evoCnt += evoCnt + wSize;
	      
	      for (n = 0; n < 16; n++)
		{
		  rval = fcw * vecSum4d[n];
		  
		  gapSzFile << fmtDbleSci(rval, 8, 16) << X2;
		}

	      gapSzFile << endl;
		 
	      for (k = 0; k < Nq; k++)
		{
		  for (n = 0; n < ntm; n++)
		    {
		      CMT_SqxWVec[k][n] += SqxWVec[k][n];
		      CMT_SqyWVec[k][n] += SqyWVec[k][n];
		      CMT_SqzWVec[k][n] += SqzWVec[k][n];
		    }
		}

	      for (n = 0; n < ntm; n++)
		{
		  CMT_qSStVec[n] += qSStVec[n];
		}	    	    
	    }
	    
	    //XXXXXXXXXX
	    //XXXXXXXXXX
	    
      /*------------------------
	Close input/output files */

      if (iAmRoot)
	{
	  gapSzFile.close();
	}
	
//XXXXXXXXXX subrouts
//XXXXXXXXXX

void get_dynSpectrum(int procNum,
		     double  *tWinVec, double **spinField,
		     double **SqxWVec, double **SqyWVec,
		     double **SqzWVec, double  *qSStVec,
		     double (&gpSzVec)[16])
	
  //--------------------------------
  // Extract frequencies associated
  // with maximum spectrum amplitude:
  
  double **specField, wfreq;

  double **xySpecArray, **yzSpecArray;
  double **zzSpecArray, **ttSpecArray;

  Vec4d xySVal0, xySVal; // 4-components vectors with
  Vec4d yzSVal0, yzSVal; // values at the 1BZ points:
  Vec4d zzSVal0, zzSVal; // K, Mx, My, G (respectively)
  Vec4d ttSVal0, ttSVal;
  
  specField = Alloc_dble_array(Nq, 3);
  
  xySpecArray = Alloc_dble_array(Lsz, Lsz);
  yzSpecArray = Alloc_dble_array(Lsz, Lsz);
  zzSpecArray = Alloc_dble_array(Lsz, Lsz);
  ttSpecArray = Alloc_dble_array(Lsz, Lsz);
	
  xySVal0 = {0.0, 0.0, 0.0, 0.0};
  yzSVal0 = {0.0, 0.0, 0.0, 0.0};
  zzSVal0 = {0.0, 0.0, 0.0, 0.0}; 
  ttSVal0 = {0.0, 0.0, 0.0, 0.0};

  for (n = 0; n < npw; n++)
    {
      for (k = 0; k < Nq; k++) // Get spectrum slice;
	{
	  specField[k][0] = SqxWVec[k][n];
	  specField[k][1] = SqyWVec[k][n];
	  specField[k][2] = SqzWVec[k][n];		  
	}
	  
      get_OrderedSpecArray2D(Lsz, specField,
			     xySpecArray, yzSpecArray,
			     zzSpecArray, ttSpecArray);

      get_specAmp(Lsz, xySpecArray, xySVal);
      get_specAmp(Lsz, yzSpecArray, yzSVal);
      get_specAmp(Lsz, zzSpecArray, zzSVal);
      get_specAmp(Lsz, ttSpecArray, ttSVal);

      wfreq = n * dwf;
      
      for (k = 0; k < 4; k++)
	{
	  i = 4 * k;
	  
	  if (xySVal0[k] < xySVal[k])
	    {
	      xySVal0[k] = xySVal[k];

	      gpSzVec[i + 0] = wfreq;
	    }
	  
	  if (yzSVal0[k] < yzSVal[k])
	    {
	      yzSVal0[k] = yzSVal[k];

	      gpSzVec[i + 1] = wfreq;
	    }
	  
	  if (zzSVal0[k] < zzSVal[k])
	    {
	      zzSVal0[k] = zzSVal[k];

	      gpSzVec[i + 2] = wfreq;
	    }
	  
	  if (ttSVal0[k] < ttSVal[k])
	    {
	      ttSVal0[k] = ttSVal[k];

	      gpSzVec[i + 3] = wfreq;
	    }
	}///| Replace with larger values;
    }
  
  deAlloc_dble_array(specField, Nq, 3);

  deAlloc_dble_array(xySpecArray, Lsz, Lsz);
  deAlloc_dble_array(yzSpecArray, Lsz, Lsz);
  deAlloc_dble_array(zzSpecArray, Lsz, Lsz);
  deAlloc_dble_array(ttSpecArray, Lsz, Lsz);
	    
	    
