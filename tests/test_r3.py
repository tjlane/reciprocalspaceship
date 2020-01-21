import unittest
import numpy as np
from os import listdir
from os.path import dirname,abspath
import re
import reciprocalspaceship as rs

try:
    from tqdm import tqdm
except:
    tqdm = iter


class TestSymmetryOps(unittest.TestCase):
    def test_hkl_to_asu(self):
        datadir = abspath(dirname(__file__)) + '/data/r3/'
        files = [datadir + i for i in listdir(datadir) if re.match(r'.*(?<!_p1).mtz$', i)] 
        for inFN in tqdm(files):
            x = rs.read_mtz(inFN)
            y = rs.read_mtz(inFN[:-4] + '_p1.mtz')
            y.spacegroup = x.spacegroup
            yasu = y.hkl_to_asu() 
            self.assertEqual(len(x.index.difference(yasu.index)), 0)
            self.assertEqual(len(yasu.index.difference(x.index)), 0)

            Fx    = x.loc[yasu.index, 'FMODEL'].values.astype(float) 
            Fyasu = yasu['FMODEL'].values.astype(float) 
            self.assertTrue(np.isclose(Fx, Fyasu).min())

            Phx    = x.loc[yasu.index, 'PHIFMODEL'].values.astype(float) 
            Phyasu = yasu['PHIFMODEL'].values.astype(float) 
            Sx    = Fx*np.exp(1j*np.deg2rad(Phx))
            Syasu = Fyasu*np.exp(1j*np.deg2rad(Phyasu))
            self.assertTrue(np.isclose(Sx, Syasu, rtol=1e-3).min())

if __name__ == '__main__':
    unittest.main()