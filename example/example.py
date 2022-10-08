import sys 
sys.path.append("../")
from dockbo.dockbo import DockBO


def example1(): 
    try:
        work_dir = sys.argv[1]
    except:
        work_dir = '/home/nmaus/'
    same_nanonet_pdb_antibody = work_dir + "dockbo/example/pdbs/83edaa7c-6f9f-488c-8baa-55c92d52dce1_nanonet_bb.pdb"


    oracle = DockBO(
        work_dir=work_dir,
        verbose_config_opt=True, # print progress during BO 
        max_n_bo_steps=40, # num bo iterations, increase to get better scores 
        bsz=1, # bo bsz (using thompson sampling)
        path_to_default_antigen_pdb=work_dir + 'dockbo/example/pdbs/6obd.pdb',
        verbose_timing=True, # print out times it takes to do stuff 
        n_init=1,
    )

    insane = [6.65447998046875, 21.624645233154297, 21.163246154785156, -0.041445255279541016, 0.08782362937927246, 0.15614235401153564, 0.029917478561401367, 0.6081490516662598, 0.6223801374435425, 0.7779067754745483, 0.7414671778678894, 0.7349568605422974, 0.7380795478820801, 0.37001341581344604, 0.4000628590583801, 0.6138855218887329, 0.03490903973579407, 0.6772670745849609, 0.5558800101280212, 0.25300168991088867, 0.5940059423446655, 0.030914923176169395, 0.2714921534061432, 0.7528858184814453, 0.5723578333854675, 0.6440922021865845, 0.26767516136169434]
    expected_score = 2095398.25
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None,
        path_to_antibody_pdb=same_nanonet_pdb_antibody, # option to pass in antibody pdb file directly, otherwise we fold seq
        config_x=insane, # ay len(27) iterable 
    )
    print(f'Score={score}, and should be: {expected_score}\n')

    # so it seems the scoring is the same, yay! 
    test_config = (19.2995851, -17.5295151, 10.1282317, -0.1058407, -0.4849369,  0.5997430, -0.6276482,  2.7736990, -1.6509974, -3.9162764,  5.5488782,  0.2823944,  0.8410559,  2.6587633,  1.8208157, -0.3644232,  3.5665332,  6.5025726,  3.2438992,  1.8170209,  6.2646740,  6.4177680,  9.5470002,  6.3438406,  3.7710937,  2.8047110,  5.9905765)    
    expected_score = -1481.95113024
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None,
        path_to_antibody_pdb=same_nanonet_pdb_antibody, # option to pass in antibody pdb file directly, otherwise we fold seq
        config_x=test_config, # ay len(27) iterable 
    )
    print(f'Score={score}, and should be: {expected_score}\n')
    # Score=-1467.913818359375, and should be: -1481.95113024 (precision in nums causes diff! )

    # example instead optimizing config_x w/ TuRBO 
    print("Using TuRBO to optimize config")
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None,# "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS",
        path_to_antibody_pdb=same_nanonet_pdb_antibody, # None, # option to pass in antibody pdb file direction, otherwise we fold seq
        config_x=None,
    )
    print(f'Score={score}\n')

    # example specifying config_x
    print("Directly passing in best config x found by TuRBO on previous run")
    test_config = oracle.previous_best_config.squeeze() 
    print("config:", test_config)
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None, # "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS",
        path_to_antibody_pdb=same_nanonet_pdb_antibody, # option to pass in antibody pdb file directly, otherwise we fold seq
        config_x=test_config, # ay len(27) iterable 
    )
    print(f'Score={score}\n')

    # example passing in pdb for antibody directoy instead of aa seq 
    #   (saves time by avoiding folding)
    print(f'Using TuRBO to optimize config_x, but with different antibody (from pdb)')
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq=None, # no need to pass in an aa seq if we instead give a .pdb file 
        path_to_antibody_pdb=same_nanonet_pdb_antibody,  # example_antibody_pdb, 
        config_x=None,
    )
    print(f'Score={score}\n')

    import pdb 
    pdb.set_trace() 

    # Expected Output:  (can vary due to BO randomnes)
    # w/ 20 BO steps: 
    # Passing in aa seq, Time for oracle call:41.79986333847046, score:305244.75
    # Passing in antibody pdb, Time for oracle call:39.276257276535034, score:2180718.75


# Timing breakdown before: 
# time to prep ligand = antigen: 47.57 
# time to prep receptor = antibody: 5.17 
# time to get adapter: 28.66
# time to get indicies: 0.11
# time to get coords: 0.000225
# time to get optimize config: 5.28 (TuRBO)
# __________________________________________________________
# Timing breakdown after w/ precomputing of antigen/ligand stuff: 
# time to prep ligand = 0 (down form 47.7 s)
# time prep receptor = 4.50 
# time get adapter = 3.60   (down from 28 s ) !! 
# time get indicies = 0.00223  (down from 0.11)
# time get coords = 0.00036 
# TOTAL TIME PREP = 8.10 s 


# __________________________________________________________
# __________________________________________________________
# __________________________________________________________
# Time to complete oracle init: 73.74

# Passing in dummy config x of all zeros
# time to fold w/ nanonet: 12.65 
# prep ligthdock time: 4.90
# compute score time: 6.12
# time for full oracle call: 23.66
# Score=2,182,445.5

# Using TuRBO to optimize config
# time to fold w/ nanonet: 10.776928901672363
# prep ligthdock time: 2.450546979904175
# compute score time: 11.220432996749878
# time for full oracle call: 24.45
# Score=328775.9375

# Directly passing in best config x found by TuRBO on previous run
# time to fold w/ nanonet: 11.057170152664185
# prep ligthdock time: 3.4004077911376953
# compute score time: 0.0509
# time for full oracle call: 14.51
# Score=132.73

# Using TuRBO to optimize config_x, but with different antibody (from pdb)
# prep ligthdock time: 3.7935421466827393
# compute score time: 15.263193368911743
# time for full oracle call: 19.06535768508911
# Score=1551769.88 



# __________________________________________________________
# TIMING SUMMARY: 
# time to fold w/ nanonet: 10-25 s 
# prep ligthdock time: 3-9 s 
# compute score time (with bad given x): 10-12 s 
# compute score time (with good given x): 0.05-1 s 
# compute score time (with TuRBO): 11-16 s (depending on N steps, this assumes 20 steps)
# time for full oracle call: 15-42 s

def example2(): 
    try:
        work_dir = sys.argv[1]
    except:
        work_dir = '/home/nmaus/'
    same_nanonet_pdb_antibody = work_dir + "dockbo/example/pdbs/83edaa7c-6f9f-488c-8baa-55c92d52dce1_nanonet_bb.pdb"

    oracle = DockBO(
        work_dir=work_dir,
        verbose_config_opt=True, # print progress during BO 
        max_n_bo_steps=500, # num bo iterations, increase to get better scores 
        bsz=1, # bo bsz (using thompson sampling)
        path_to_default_antigen_pdb=work_dir + 'dockbo/example/pdbs/pdb1e4j.pdb',
        verbose_timing=False, # print out times it takes to do stuff 
        n_init=1,
    )

    # example instead optimizing config_x w/ TuRBO 
    print("Using TuRBO to optimize config")
    score = oracle(
        path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
        antibody_aa_seq="TVSSTVSSTVSSTVSSTVSSDKAKGYTTETVSSTVSSTVSSTVYNPTVSSTVSSTVSSTVSVKGRVTVSSTVSSTVSSTVTMLVDTSTVSSTVSSTVSSTVKNQFSLTVSSTVSSTVSSYWGQGSTVRTVSSTVSSTVSSTVLSSVTATVSSTVAAPFDLSSTVSSTVADTAVYYCARTVSSTVSSTVSSTVEGHTVTVSS",
        path_to_antibody_pdb=None, # None, # option to pass in antibody pdb file direction, otherwise we fold seq
        config_x=None,
    )
    print(f'Score={score}\n')


def example3(): # test dfire 
    try:
        work_dir = sys.argv[1]
    except:
        work_dir = '/home/nmaus/'
    same_nanonet_pdb_antibody = work_dir + "dockbo/example/pdbs/e4ae16c0-6d10-40b6-9bb7-b5ed96447843_nanonet_bb.pdb"
    # reg dfire: 2b122d44-12fa-4011-beae-60b8ba25f94b_nanonet_bb.pdb"   
    # fast-dfire d161ea3d-3c5d-4cc7-8567-908dd27f64e3_nanonet_bb.pdb
    antigen_path = work_dir + 'dockbo/example/pdbs/6obd.pdb'
    # flip antibody and antigen below for direct comparison to way I'm running default lightdock, works :) !
    oracle = DockBO(
        work_dir=work_dir,
        verbose_config_opt=True, # print progress during BO 
        max_n_bo_steps=500, # num bo iterations, increase to get better scores 
        bsz=1, # bo bsz (using thompson sampling)
        path_to_default_antigen_pdb=same_nanonet_pdb_antibody, 
        verbose_timing=False, # print out times it takes to do stuff 
        n_init=1,
        scoring_func='dfire2',
    )

    # w default --> fast lightdock ... BAD! (unsure why)
    # expected_scores = [22.25233176,  -104.55805133, -5.73553140, 2.16423355, 7.04462614] 
    # test_x1 = (-18.4669302, 28.2081904, -12.9068881,  0.0518200, -0.3179082, -0.3013085, -0.8974755,  4.6604336,  3.7999389,  0.9439976,  9.4457524,  2.2482698,  4.5980431,  2.5652234,  4.0266471,  8.3063899,  0.0221808, -0.9783754,  3.8438233,  1.7701144,  6.1862887,  6.9060350,  3.2899047,  8.5243417,  3.1163853,  1.7386714,  2.4620691)
    # test_x2 = (-17.6972587, -17.2697541, -30.9130148,  0.6756919,  0.3221647,  0.3861439,  0.5390206,  0.8903689,  2.9584027, -0.0652793,  2.6138636,  4.8109099,  7.2268635,  5.7851981,  4.1365822,  5.4829824,  2.5112768,  6.9890948,  2.7226356,  5.0510690,  7.7267428,  3.8577612, 10.0304599,  5.8676770,  3.5265084,  4.6436507, -0.7836380) 
    # test_x3 = (-24.6077525, -5.5872635, -32.2215955,  0.4818323,  0.6904685, -0.5371949, -0.0501244,  3.8759446,  5.7363090,  9.9249166,  3.8584816,  7.1619745,  4.6923882,  5.2178137,  3.1646439,  6.9532388,  7.9266247,  4.2034769,  3.2995185, -1.4668570,  8.0139417,  4.2778450,  3.1877427, -2.6555378,  3.7120215,  2.7967108,  3.2535146)
    # test_x4 = (-23.7635989, -7.3345546, -35.8217536, -0.4674992,  0.5679796, -0.4982231, -0.4589308,  3.2723340,  2.6708834, -0.2732658,  4.2918862,  6.1024956,  4.3471042,  7.6592775,  6.2649717,  1.8584968,  8.1654184,  6.9438062,  3.2054680,  4.7697911,  0.6669195,  2.0253651,  6.0387414,  2.0301505,  5.2488538, -2.3637364,  2.1527191) 
    # test_x5 = (-19.5652820, -14.2492003, -40.3244981, -0.8314282,  0.0581858, -0.5296326,  0.1575780,  3.4059079,  9.3261203,  4.4186365,  4.0620732,  8.3683999,  4.8147847,  4.8204160,  2.8578700,  9.0068094,  4.0826098,  0.6668608,  5.0554656,  1.9270701,  5.6611197,  0.2746346,  3.8886622,  5.0771423,  4.0297325,  4.2914704,  4.4529382) 
    
    # reg defire: BAD! (unsure why)
    # test_x6 = (-38.0617934, -15.5883233, 12.1150632, -0.8895526, -0.1112063, -0.4098987, -0.1682628, 10.3989383,  2.5706245,  0.5217251,  6.0040070,  2.3781107,  3.0959363,  3.9240965,  1.7617086,  6.5100443,  0.7843271,  1.3066368,  7.2032634,  4.2286843,  4.5302404,  4.6142456,  1.6326213,  2.6557925,  9.0471881, -0.4369314, 11.6552815) 
    # test_x1 = (21.9925472, -12.9950009, 23.0394955, -0.5257622,  0.0415576, -0.8454891, -0.0836377,  5.7573432,  4.6256818,  4.6483175,  5.1562806,  5.5587723,  3.4552138,  9.7995838,  5.1130890,  4.6518012, -0.2975063,  2.4124260,  1.9186426,  5.3390069,  5.6068900,  1.6401877,  6.8399505,  3.8950816,  3.7891448,  5.1866952,  2.5087438)    
    # expected_scores = [ -49.07526396, 10.92530477, 7.97403700, -105.31586891, -2.61792350,  3.58476832] 
    # test_x2 = (21.7676490, -24.0939427, 18.7655656, -0.3566682, -0.0065375,  0.3012389,  0.8843077,  4.0432557,  5.3047560,  6.8475974,  1.6180896,  4.7716222,  0.6981379,  4.3091603,  0.9707273,  4.0550270,  4.5482110,  0.7867277,  9.0723786,  5.7627480,  7.8734011,  2.2651359,  5.9079430,  6.9942473,  8.6632339,  3.2714386,  8.2227488)    
    # test_x3 = ( 9.6589372, 15.4681349, -34.5456028, -0.1853718, -0.7573551, -0.3279770,  0.5333682, -1.0803271,  4.3366254,  2.9010967,  1.1219413,  5.8761193,  5.8524998, -1.4456569,  4.3456765,  9.0379236,  4.6815427,  1.2040545,  3.9398004,  0.5468150,  6.8734155,  1.4282478,  6.3306072,  5.7164022,  3.3016714,  3.7333875,  7.0732332)    
    # test_x4 = ( 4.7328460,  7.9698244, -28.0926581,  0.2195080,  0.6282110,  0.0188731,  0.7461976,  6.0882041, -1.2539899, 11.5830306,  2.3471072,  5.4585562,  0.3750677,  5.8866869,  5.4434497,  2.5375002,  4.4561225,  9.8342036,  0.4324131,  3.8542895,  6.3381459,  5.5891916,  6.7633092,  5.2501005,  1.0413792,  3.9811362,  3.9779061)    
    # test_x5 = (-33.3224630, -12.8037865, 15.4302590,  0.5385880,  0.2697013,  0.5613560, -0.5675065,  2.0033799,  8.5547086,  2.2305011,  7.7178546,  2.8585025,  5.0459282,  0.1827396,  4.4842253,  5.4429092,  4.9668096,  7.7979941,  5.1204619, -0.8431540,  6.6892530,  2.3270920,  2.9037190,  6.3382907,  2.7819490,  8.7433981,  4.2210703) 
    # 

    # dfire2:  GOOD! (correlated w/ negative of real score) --> remains to correlate w/ lightdock! 
    test_x1 = (22.7351045, -18.9733075, 15.1090247,  0.3972159,  0.4494580, -0.7733119,  0.2054160,  3.3939638,  3.1163392,  8.9033879,  9.4865672,  5.2258604,  3.8064031,  5.6948983,  1.1246080,  4.6297169,  2.7943511,  1.7685366,  3.1158313,  6.4125272,  3.1325591,  3.6944656,  5.3991238, -0.6966649,  7.4570192,  0.5318471,  0.6353195)    
    e1 = -1534.17302396
    test_x2 = (19.2995851, -17.5295151, 10.1282317, -0.1058407, -0.4849369,  0.5997430, -0.6276482,  2.7736990, -1.6509974, -3.9162764,  5.5488782,  0.2823944,  0.8410559,  2.6587633,  1.8208157, -0.3644232,  3.5665332,  6.5025726,  3.2438992,  1.8170209,  6.2646740,  6.4177680,  9.5470002,  6.3438406,  3.7710937,  2.8047110,  5.9905765)    
    e2 = -1481.95113024
    test_x3 = (20.6730378, -21.3924403, 21.1683670,  0.1462513, -0.9300613,  0.2960356,  0.1611194,  0.2247729,  1.6623545,  8.0687043, -1.3403247,  7.5234849,  7.4086563,  5.0197114,  2.5118978,  7.8305798,  7.9994238,  0.3458648,  6.4869254, -0.2510850,  9.4316685, -2.1054539,  1.0177214,  5.0451083,  0.0184717,  5.6433976,  7.6885780)    
    e3 = -1535.28530623
    test_x4 = (-31.4431133,  5.1388721,  3.2096625, -0.1754744, -0.8343950, -0.5221402, -0.0190587, 10.3997868,  6.2620204,  4.2348898,  8.7148827, -2.2442208,  6.9104260,  1.9029248,  0.3025138,  7.5659565,  5.5772166,  5.5129993,  4.4938171,  2.7166056,  0.8173575, -0.6120374,  4.7650628,  4.5091405, -3.9799379,  4.3115087,  6.6716461)    
    e4 = -1489.42750471
    test_x5 = (-26.7321482,  4.3626478,  2.1799333,  0.3516821,  0.5774817, -0.2642751, -0.6877450,  5.4467544,  5.8166520, -4.8175574,  9.9438096,  1.4902349,  8.6398457,  6.6309471,  3.2796301,  2.6751565,  0.6612416,  2.0444718, -0.1430187,  3.4174874,  2.2652036,  9.2017723,  4.3929141,  2.0534601,  8.4318878,  7.9883808,  3.2340427)  
    e5 = -1495.45363480

    expected_scores = [e1, e2, e3, e4, e5]
    test_xs = [test_x1, test_x2, test_x3, test_x4, test_x5]
    for test_x, expected_score in zip(test_xs, expected_scores):
        score = oracle(
            path_to_antigen_pdb=None, # no antigen path specified --> use default (aviods recomputing default structure)
            antibody_aa_seq=None, 
            path_to_antibody_pdb=antigen_path, # same_nanonet_pdb_antibody, # None, # option to pass in antibody pdb file direction, otherwise we fold seq
            config_x=test_x,
            # enforce_no_convex_hull_overlap=False,
        )
        print(f'Score={score}, and should be: {expected_score}\n')


    # fast dfire 
    # Score=22.577552795410156, and should be: 22.25233176
    # Score=-114.30744171142578, and should be: -104.55805133
    # Score=-9.363850593566895, and should be: -5.7355314

    # reg dfire: 
    # Score=-47.38373565673828, and should be: -49.07526396 
    # Score=10.834867477416992, and should be: 10.92530477
    # Score=-6.09305477142334, and should be: 7.974037
    # Score=-107.02908325195312, and should be: -105.31586891
    # Score=-17.021217346191406, and should be: -2.6179235 
    # Score=-5.836091995239258, and should be: 3.58476832

    # dfire2 
    # Score=1527.3583984375, and should be: -1534.17302396
    # Score=1467.913818359375, and should be: -1481.95113024
    # Score=1510.666015625, and should be: -1535.28530623
    # Score=1491.0162353515625, and should be: -1489.42750471
    # Score=1488.0440673828125, and should be: -1495.4536348
    

if __name__ == "__main__":
    # example1() 
    example3() 
