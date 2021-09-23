# import libraries
import numpy as np
import pandas as pd
import mplhep as hep
import uproot_methods.classes.TLorentzVector as LVepm

from numpy import pi, sqrt, sin, cos, linspace, zeros, arctan, exp, tan, arccos

from numpy import array as arr
from numpy import append as app
from pandas import DataFrame


# function to split dataframe into 4, only for eeuu
def partner_eeuu(df: DataFrame=None):
    cols = [col for col in df.columns]

    s = ( len(df), len(cols) )
    a, b, c, d = zeros( shape=s ), zeros( shape=s ), zeros( shape=s ), zeros( shape=s )

    for i, row in enumerate(df.iloc):
        x = np.stack( row[:-4].to_numpy() )
        
        x_add = row[-4:].to_numpy()
        
        # ab: electrons, cd: muons
        ai, bi = tuple( np.where( x[cols.index('type')] == 11 )[0] )
        ci, di = tuple( np.where( x[cols.index('type')] == 13 )[0] )

        x = np.transpose(x)

        if  x[ai][cols.index('Q')] == 1:
            ai, bi = bi, ai
        if x[ci][cols.index('Q')] == 1:
            ci, di = di, ci
        
        
        a[i], b[i] = app(x[ai], x_add), app(x[bi], x_add)
        c[i], d[i] = app(x[ci], x_add), app(x[di], x_add)
    
    e2u2 = [ cart( pd.DataFrame(dic, columns=cols) ) for dic in [a, b, c, d] ]
    cols = e2u2[0].columns
    sorter = mass_compare_eeuu(frames=e2u2)
    
     # add sorter to each array
    for df in e2u2:
        df['s'] = sorter
        
    # get lepton dataframes such that AB real and CD virtual
    A = pd.concat( [ e2u2[0][sorter == True], e2u2[2][sorter == False]  ] )
    B = pd.concat( [ e2u2[1][sorter == True], e2u2[3][sorter == False]  ] )
    C = pd.concat( [ e2u2[2][sorter == True], e2u2[0][sorter == False]  ] )
    D = pd.concat( [ e2u2[3][sorter == True], e2u2[1][sorter == False]  ] )
    
    out = [A, B, C, D]
    
    # order dataframes, such that they align, and drop sorter column
    for df in out:
        df.sort_values(by='id', axis=0, inplace=True)
        df.drop(labels=['s'], axis=1, inplace=True)
    
    return out

# a function to compare eeuu set, and return sorter such that AB is real and CD virtual
def mass_compare_eeuu(frames: list=None, zPDG: float=91188):
    frames = [ df[ ['E', 'px', 'py', 'pz'] ] for df in frames ]
    
    names = ['ab', 'cd']
    frames = [ frames[0] + frames[1],
               frames[2] + frames[3] ]
    dic = {}
    for i, df in enumerate(frames):
        dic[names[i]] = abs( sqrt( df['E']**2 - df['px']**2 - df['py']**2 - df['pz']**2 ) - zPDG )
    df = pd.DataFrame(dic)
    
    # true means that AB is the real boson and CD the virtual
    return df['ab'] <= df['cd']

# split function for eeee or uuuu, partners lepton's to prioritize Z-mass for single pair
def partner_llll(df: DataFrame=None):
    cols = [col for col in df.columns]

    # create containers for initial split, lower-case to avoid confusion with final split
    s = ( len(df), len(cols) )
    a, b, c, d = zeros( shape=s ), zeros( shape=s ), zeros( shape=s ), zeros( shape=s )

    for i, row in enumerate(df.iloc):
        x = np.stack( row[:-4].to_numpy() )
        
        x_add = row[-4:].to_numpy()
        
        # ab: electrons (muons), cd: positrons (antimuons)
        ai, bi = tuple( np.where(x[cols.index('Q')] == -1 )[0] )
        ci, di = tuple( np.where(x[cols.index('Q')] == 1 )[0] )
        
        x = np.transpose(x)
        a[i], b[i] = app(x[ai], x_add), app(x[bi], x_add)
        c[i], d[i] = app(x[ci], x_add), app(x[di], x_add)
    
    # pass into mass_compare function to get a boolean sorter
    L4 = [ cart( pd.DataFrame(dic, columns=cols) ) for dic in [a, b, c, d] ]
    cols = L4[0].columns
    sorter = mass_compare_llll(frames=L4)
    
    # add sorter to each array
    for df in L4:
        df['s'] = sorter
        
    # get lepton dataframes such that A partners with B, C partners with D
    A = pd.concat( [ L4[0][sorter == True], L4[1][sorter == False]  ] )
    B = pd.concat( [ L4[2][sorter == True], L4[3][sorter == False]  ] )
    C = pd.concat( [ L4[1][sorter == True], L4[0][sorter == False]  ] )
    D = pd.concat( [ L4[3][sorter == True], L4[2][sorter == False]  ] )
    
    out = [A, B, C, D]
    
    # order dataframes, such that they align, and drop sorter column
    for df in out:
        df.sort_values(by='id', axis=0, inplace=True)
        df.drop(labels=['s'], axis=1, inplace=True)
    
    return out

# a function to compare 4 similar leptons and return a series of booleans
def mass_compare_llll(frames: list=None, zPDG: float=91188):
    frames = [ df[ ['E', 'px', 'py', 'pz'] ] for df in frames ]
    
    # assume input is [a, b, c, d], then this list gives the matching which we test
    names = ['ac', 'ad', 'bc', 'bd']
    frames = [ frames[0] + frames[2], 
               frames[0] + frames[3], 
               frames[1] + frames[2], 
               frames[1] + frames[3] ]
    
    # calculate Z-mass diff for each of the above combinations
    dic = {}
    for i, df in enumerate(frames):
        dic[names[i]] = abs( sqrt( df['E']**2 - df['px']**2 - df['py']**2 - df['pz']**2 ) - zPDG )
    df = pd.DataFrame(dic)
    
    # if ac then bd, and we only need 1 Z-boson close to 91 GeV
    df['ac'] = df[ ['ad', 'bd'] ].min(axis=1)
    df['ad'] = df[ ['ad', 'bc'] ].min(axis=1)
    df.drop(labels=['bd', 'bc'], axis=1, inplace=True)
    
    # return sorted array, (ac, bd) = True, (ad, bc) = False
    return df['ac'] <= df['ad']

# replace ptetaphi with cartesian coordinates
def cart(df: DataFrame=None):
    theta = 2*arctan( exp( -df['eta'] ) )
    df['px'] = df['pt'] * sin( df['phi'] )
    df['py'] = df['pt'] * cos( df['phi'] )
    df['pz'] = df['pt'] / tan(theta)
    df['p'] = sqrt( df['px']**2 + df['py']**2 + df['pz']**2 )
    df.drop(labels=['pt', 'eta', 'phi'], axis=1, inplace=True)
    return df

# Merge and Mass, a generalized function, either LLLL -> ZZ, or ZZ -> H 
def MnM(FRAMES: list=None):
    # if input is LLLL, return ZZ
    if len(FRAMES) == 4:
        frames = [df.drop(labels=['p'], axis=1, inplace=False) for df in FRAMES]
        X, Y = frames[0] + frames[1], frames[2] + frames[3]
        X['id'], Y['id'] = X['id']/2, Y['id']/2
        X['w'], Y['w'] = X['w']/2, Y['w']/2

        out = []
        for i, df in enumerate( [X, Y] ):
            df['p'] = sqrt( df['px']**2 + df['py']**2 + df['pz']**2 )
            df['m'] = sqrt( df['E']**2 - df['p']**2 )
            out.append( df )

        return out
    
    # if input is ZZ, return H
    elif len(FRAMES) == 2:
        df = FRAMES[0] + FRAMES[1]
        df['id'] = df['id']/2
        df['w'] = df['w']/2
        df.drop(labels=['p'], axis=1, inplace=True)
        if 'm' in df.columns:
            df.drop(labels=['m'], axis=1, inplace=True)
        df['p'] = sqrt( df['px']**2 + df['py']**2 + df['pz']**2 )
        df['m'] = sqrt( df['E']**2 - df['p']**2 )
        
        return df
    
 # momentum dot product between 2 particles
def dot(d1: DataFrame=None, d2: DataFrame=None):
    return d1['px']*d2['px'] + d1['py']*d2['py'] + d1['pz']*d2['pz']

# get angles between lepton and Z-boson, as well as angle between Z and higgs
def get_angles(llll: list=None, zz: list=None, h: DataFrame=None):
    # dict for mapping lep to z
    L2Z = {0: 0, 1: 0, 2: 1, 3: 1}
    
    # 'a' for angle :)
    for i, l in enumerate(llll):
        z = zz[ L2Z[i] ]
        l['a'] = arccos( dot(l, z) / (l['p']*z['p']) )
    for i, z in enumerate(zz):
        z['a'] = arccos( dot(z, h) / (h['p']*z['p']) )
    
    return llll, zz, h

# merge all the levels into a single dataframe
def merge(llll: list=None, zz: list=None, h:list=None):
    
    # rename and merge leptons
    for i, df in enumerate(llll):
        df.sort_values(by='id', axis=0, inplace=True)
        df.drop(labels=['id', 'w', 'sig?', 'fam'], axis=1, inplace=True)
        dic = [f'L{i} '+col for col in df.columns]
        dic = dict( zip(df.columns, dic) )
        df.rename(mapper=dic, axis=1, inplace=True)
    llll = pd.concat(llll, axis=1)
    
    # rename and merge Z's
    for i, df in enumerate(zz):
        df.sort_values(by='id', axis=0, inplace=True)
        df.drop(labels=['w', 'sig?', 'id', 'fam'], axis=1, inplace=True)
        dic = [f'Z{i} '+col for col in df.columns]
        dic = dict( zip(df.columns, dic) )
        df.rename(mapper=dic, axis=1, inplace=True)
    zz = pd.concat(zz, axis=1)
    
    h.sort_values(by='id', axis=0, inplace=True)
    out = pd.concat([llll, zz, h], axis=1)
    return out

# a combination of all above process, returns informative dataframe if export=False, else exports 
def pipeline(df: DataFrame=None, subset: str='All', export: bool=True):
    
    if subset == 'all':
        typesum = [52, 44, 48]
    elif subset == 'eeuu':
        typesum = [48]
    elif subset == 'eeee':
        typesum = [44]
    elif subset == 'uuuu':
        typesum = [52]
    elif subset == 'llll':
        typesum = [52, 44]

    leps = [df[df['sum_lep_type']==ts].drop(labels=['sum_lep_type'], 
                                            axis=1, 
                                            inplace=False) for ts in typesum]

    frames = []
    for i, lep in enumerate(leps):

        # split and match leptons
        if typesum[i] == 48:
            llll = partner_eeuu(lep)
        else:
            llll = partner_llll(lep)

        # get Z's and higgs
        zz = MnM(llll)
        h = MnM(zz)

        # append to list of dataframes
        frames.append( merge( *get_angles(llll, zz, h) ) )

    out = pd.concat(frames, axis=0)
    out.sort_values(by='id', axis=0, inplace=True)
    out.set_index(keys='id', drop=True, inplace=True)
    if export:
        out.to_csv('DataFrame.csv')
    else:
        return out