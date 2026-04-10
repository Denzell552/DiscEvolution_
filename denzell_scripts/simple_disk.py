import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('/Users/denzell/Documents/W26_Coop/DiscEvolution_/') 

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler
import h5py

from DiscEvolution.constants import *
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import *
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution, HybridWindModel, TaboneSolution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import *
from DiscEvolution.dust import PlanetesimalFormation
from DiscEvolution.planet import Planet
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.chemistry import *
from copy import deepcopy

import time
start_time = time.time()


def run_model(config):

    """
    Run the disk evolution model and plot the results
    
    Parameters:
    config(dict): configuration dictionary containing all perameters

    """
    # Extract parameters form config dictionary
    grid_params = config['grid']
    sim_params = config['simulation']
    star_params = config['star']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planets']
    chemistry_params = config["chemistry"]
    planetesimal_params = config['planetesimal']
    wind_params = config['winds']
    gap_params = config['gap']

    # Set up disc
    # ==============================

    # Create the grid
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])

    # Create star
    star = SimpleStar(M = star_params["M"], R = star_params["R"], T_eff = star_params['T_eff'])


    # Create time array
    if sim_params['t_interval'] == "power":
        # Determine the number of points needed
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            times = np.logspace(0, np.log10(sim_params['t_final']), num=num_points) * 2 * np.pi
        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            times = np.logspace(np.log10(sim_params['t_initial']), np.log10(sim_params['t_final']), num=num_points) * 2 * np.pi
    elif type(sim_params['t_interval']) == list:
        times = np.array(sim_params['t_interval']) * 2 * np.pi * 1e6
    else:
        times = np.arange(sim_params['t_initial'], sim_params['t_final'], sim_params['t_interval']) * 2 * np.pi
   
   
    # define opacity class used. If not Tazzari, defaults to Zhu in IrradiatedEOS.
    if eos_params["opacity"] == "Tazzari":
        kappa = Tazzari2016()
    else:
        kappa = None


    # Create initial Sigma profile based on desired disc parameters
    # ======================================================
   
    if grid_params['type'] == 'Booth-alpha':
        # For fixed Rd, Mdot and Mdisk, solve for alpha
    
        # extract params
        Mdot=disc_params['Mdot']
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        Rd=disc_params['Rd']
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd/R) * np.exp(-R/Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)
            return Sigma
    
        # define an initial guess for the Sigma profile
        Sigma = Sigma_profile(R, Rd, Mdisk)
    
        # define a gas class, to be used later 
        gas_temp = ViscousEvolutionFV()

        # iterate to get alpha
        for j in range(100):
            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
            # update eos with current sigma profile
            eos.set_grid(grid)
            eos.update(0, Sigma)

            # define a disc given current eos and Sigma
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find the current Mdot in the disc
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, Sigma))

            # scale Sigma by Mdot to get desired Mdot.
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define new disc given new Sigma profile
            disc = AccretionDisc(grid, star, eos, Sigma)

            # scale alpha by Mdisk so that desired disk mass is achieved.
            alpha= alpha*(disc.Mtot()/Msun)/Mdisk

            if grid_params["smart_bining"]:
                # if using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

        alpha_SS = alpha

    elif grid_params['type'] == 'Booth-Rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd
    
        # extract params
        Mdot=disc_params['Mdot']
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        Rd=disc_params['Rd'] # initial guess
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd/R) * np.exp(-R/Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)
            return Sigma
    
        # create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
        # update eos with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        # define gas classe to be used in first iteration
        gas_temp = ViscousEvolutionFV()

        # iterate to get Rd
        for j in range(100):
            # initialize a disc with current Sigma and eos
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S=Sigma))

            # Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new= Rd*np.sqrt(Mdisk/(disc.Mtot()/Msun))
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            # define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            # update eos with new Sigma to have correct temperature profile
            eos.update(0, Sigma)

        alpha_SS = alpha

    elif grid_params['type'] == "LBP":
        # define viscous evolution to calculate drift velocity later
        gas = ViscousEvolutionFV()

        # extract parameters
        gamma=disc_params['gamma']
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot']* Msun/yr 
        Mdisk=disc_params['M']* Msun
        alpha=disc_params['alpha']
        mu=chemistry_params['mu']
        rin=R[0]
        xin=R[0]/Rd

        # calculate the keplerian velocity
        fin=np.exp(-xin**(2.-gamma))*(1.-2.*(2.-gamma)*xin**(2.-gamma))
        nud_goal=(Mdot/Mdisk)*(2.*Rd*Rd)/(3.*(2.-gamma))/fin*AU*AU #cm^2
        nud_cgs=nud_goal*yr/3.15e7
        Om_invsecond=star.Omega_k(Rd)*yr/3.15e7

        # calculate initial sound speed and temperature profile
        cs0 = np.sqrt(Om_invsecond*nud_cgs/alpha) #cm/s
        Td=cs0*cs0*mu*m_p/k_B #KT=Td*(R/Rd)**(gamma-1.5)
        T=Td*(R/Rd)**(gamma-1.5)

        # calculate the actual sound speed and surface density profile
        cs = np.sqrt(GasConst * T / mu) #cgs
        cs0 = np.sqrt(GasConst * Td / mu) #cgs
        nu=alpha*cs*cs/(star.Omega_k(R)*yr/3.15e7) # cm2/s
        nud=np.interp(Rd,grid.Rc,nu)*3.15e7/yr # cm^2 
        Sigma=LBP_Solution(Mdisk,Rd*AU,nud,gamma=gamma)
        Sigma0=Sigma(R*AU,0) 

        # Adjust alpha so initial Mdot is correct
        for i in range(10):
            # define an EOS
            eos = IrradiatedEOS(star, alpha_t=disc_params['alpha'], kappa=kappa, Tmax=eos_params["Tmax"])
            eos.set_grid(grid)
            eos.update(0, Sigma0)

            # define a temporary disc to compute Mdot
            disc = AccretionDisc(grid, star, eos, Sigma0)

            # adjust alpha depending on current Mdot and wanted Mdot
            vr=gas.viscous_velocity(disc,Sigma0)
            Mdot_actual=disc.Mdot(vr[0])#* (Msun / yr)
            alpha=alpha*(Mdot/Msun*yr)/Mdot_actual
        Sigma = Sigma0

    elif grid_params['type'] == 'Booth-Mdot':
        # For fixed alpha, Rd, and Mdisk, solve for Mdot
    
        # extract parameters
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot']* Msun/yr # initial guess
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        Sigma = (Rd/R) * np.exp(-R/Rd)
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)

        alpha_SS = alpha

    elif grid_params['type'] == 'winds-alpha':
        # For fixed Rd, Mdot and Mdisk, solve for alpha with disk winds
        # assumes gamma = 1

        # extract params
        Mdot=disc_params['Mdot'] # solar masses per year
        Mdisk=disc_params['M']* Msun
        psi = wind_params['psi_DW']
        #lambda_DW = wind_params['lambda_DW']
        Rd=disc_params['Rd']
        alpha = disc_params['alpha']
        e_rad=wind_params["e_rad"]
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha/(1 + psi)

        # initial guess for Sigma
        Sigma_d = Mdisk/(2 * np.pi * (Rd*AU)**2)
        #xi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
        xi = 0
        Sigma = Sigma_d * (R/Rd)**(xi - gamma) * np.exp(-(R/Rd)**(2 - gamma))

        # define an initial disc and gas class to be used later
        disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
        gas_temp = HybridWindModel(psi, lambda_DW)

        # scale Sigma by current Mtot just in case Sigma is not quite at the correct value to have the desired Mdisk (which often happens)
        Mtot = disc.Mtot()
        Sigma[:] *= Mdisk / Mtot

        for i in range(100):
            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad, Tmax=eos_params["Tmax"])
            
            # update eos with grid and Sigma
            eos.set_grid(grid)
            eos.update(0, Sigma)

            # define new disc 
            disc = AccretionDisc(grid,star,eos,Sigma)

            # find current Mdot in the disc given Sigma and current eos
            vr = gas_temp.viscous_velocity(disc,Sigma)
            Mdot_actual = disc.Mdot(vr)[0] # solar masses per year

            # Scale alpha by Mdot
            alpha_new = alpha*Mdot/Mdot_actual
            alpha = 0.5 * (alpha + alpha_new) # average done to damp oscillations in numerical solution

            # find a new alpha_SS given new alpha.
            alpha_SS = alpha/(1 + psi)

            if grid_params["smart_bining"]:
                # if using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

    elif grid_params['type'] == 'winds-Rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd with disk winds
    
        # extract params
        Mdot=disc_params['Mdot'] # solar masses per year
        Mdisk=disc_params['M']* Msun
        psi = wind_params['psi_DW']
        #lambda_DW = wind_params['lambda_DW']
        Rd=disc_params['Rd']
        alpha = disc_params['alpha']
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha/(1 + psi)

        def Sigma_profile(R, Rd, Mdisk):
            """Creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
            Sigma = (R/Rd)**(chi - gamma) * np.exp(-(R/Rd)**(2 - gamma))
            Sigma *= Mdisk / np.trapezoid(Sigma, np.pi * (R * AU)**2)
            return Sigma
    
        # create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, Tmax=eos_params["Tmax"])
        
        # update eos with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        # define gas classe to be used in first iteration
        gas_temp = HybridWindModel(psi, lambda_DW)

        # iterate to get Rd
        for j in range(100):
            # initialize a disc with current Sigma and eos
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S=Sigma))

            # Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new= Rd*np.sqrt(Mdisk/disc.Mtot())
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            # define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            # update eos with new Sigma to have correct temperature profile
            eos.update(0, Sigma)

    elif grid_params['type'] == 'winds-Mdot':
        # For fixed alpha, Rd, and Mdisk, solve for Mdot with disk winds included
    
        # extract parameters
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot'] # initial guess
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        psi = wind_params['psi_DW']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        gamma = disc_params['gamma']
        alpha_SS = alpha/(1+psi)

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
        Sigma = (R/Rd)**(chi - gamma) * np.exp(-(R/Rd)**(2 - gamma))
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad, Tmax=eos_params["Tmax"])
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)


    elif grid_params['type'] == 'tabone':
        # For hybrid wind model using Tabone et al. 2022

        # exctract parameters
        R = grid.Rc
        Rd = disc_params['Rd']
        Mdisk = disc_params['M'] * Msun
        psi = wind_params['psi_DW']
        d2g = disc_params['d2g']
        alpha = disc_params['alpha']
        Mdot = disc_params['Mdot'] 
        lambda_DW = 3
        alpha_SS = alpha / (1 + psi)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, Tmax=eos_params["Tmax"])

        eos.set_grid(grid)
        gas_temp = HybridWindModel(psi, lambda_DW)
        nu_c = eos._f_nu(Rd)
        tabone = TaboneSolution(Mdisk, Rd*AU, nu_c, psi, d2g, lambda_DW)

        Sigma = tabone(R*AU, 0) 

        # to calculate alpha
        # scale Sigma by current Mtot just in case Sigma is not quite at the correct value to have the desired Mdisk (which often happens)
        disc = AccretionDisc(grid,star,eos,Sigma)
        Mtot = disc.Mtot()
        Sigma[:] *= Mdisk / Mtot

        for i in range(100):

            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, Tmax=eos_params["Tmax"])

            # update eos with grid and Sigma
            eos.set_grid(grid)
            eos.update(0, Sigma)

            # update sigma profile using Tabone solution with updated eos
            nu_c = eos._f_nu(Rd)
            tabone = TaboneSolution(Mdisk, Rd*AU, nu_c, psi, d2g, lambda_DW)
            Sigma = tabone(R*AU, 0) 

            # define new disc 
            disc = AccretionDisc(grid,star,eos,Sigma)

            # find current Mdot in the disc given Sigma and current eos
            vr = gas_temp.viscous_velocity(disc,Sigma)
            Mdot_actual = disc.Mdot(vr)[0] # solar masses per year

            # Scale alpha by Mdot
            alpha_new = alpha*Mdot/Mdot_actual
            alpha = 0.5 * (alpha + alpha_new) # average done to damp oscillations in numerical solution

            # find a new alpha_SS given new alpha.
            alpha_SS = alpha/(1 + psi)



    # Set up dynamics
    # ========================
    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW)
        else:
            gas = ViscousEvolutionFV()
    
    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        dust = SingleFluidDrift(diffusion=diffuse, settling=dust_growth_params['settling'], van_leer=transport_params['van_leer'])
        diffuse = None

    # Set disc model
    # ========================
    try:
        disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], 
            Sigma=Sigma, feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
            f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
            uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas
        )
    except Exception as e:
        #disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], Sigma=Sigma, f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'])
        raise e
    
    #### Chemistry, planetessimals and planets not included but would be set up here
    disc.chem = None
    disc.planetesimals = None
    disc.planets = None

    # creating pressure gradient
    def gammaP(disc):
        '''Calculate the dimensionless non absolute pressure gradient'''

        P = disc.P
        R = grid.Rc
        gamma = np.empty_like(P)
        gamma[1:-1] = (P[2:] - P[:-2])/(R[2:] - R[:-2])
        gamma[ 0]   = (P[ 1] - P[  0])/(R[ 1] - R[ 0])
        gamma[-1]   = (P[-1] - P[ -2])/(R[-1] - R[-2])
        gamma *= R/(P+1e-300)

        return gamma


    # Set gap profile model
    # ==========================

    if gap_params['on']:
        if gap_params['type'] == 'duffell2019':
            ''' Planetary Gap: Using current gap profile from planet.py (Duffell 2019) '''
        
            planet = Planet(
                Mp=planet_params['Mp'][0],
                ap=planet_params['Rp'][0],
            )

            Mp = planet_params['Mp'][0]
            Rp = planet_params['Rp'][0]

            # gap depth array
            gap_depth = planet.gap_profile(disc)

            # apply gap profile using set_gap_profile method, preturbs the viscosity profile to induce a gap.
            # If locallyisothermalEOS is used, the viscosity profile maintains the same shape throughout the evolution.
            disc.set_gap_profile(gap_depth)

            def edge():
                '''Calculate the location of the gap edge, defined as where the gap starts to dip'''
                
                edges = []

                for i in gap_depth:
                    if i / np.max(gap_depth) < 0.97: # if gap depth is 97% of max depth, we are at the edge
                        idx = np.argmin(np.abs(gap_depth - i))
                        edges.append(grid.Rc[idx])

                idx_inner = np.argmin(np.abs(grid.Rc - edges[0]))
                idx_outer = np.argmin(np.abs(grid.Rc - edges[-1]))
    
                return edges[0], edges[-1], idx_inner, idx_outer
    
        elif gap_params['type'] == 'kanagawa2016':
            ''' Planetary Gap: Using kanagawa 2016 to set gap profile '''

            Rp = planet_params['Rp'][0]
            Mp = planet_params['Mp'][0]

            q = (Mp * Mjup) / (star_params['M'] * Msun) # planet- star mass ratio
            hp = disc.H[np.argmin(np.abs(grid.Rc - Rp))]  # scale height at Rp

            K = q**2 * (hp/Rp)**(-5) * disc._eos._alpha_t**(-1) # Kanagawa K parameter (used for gap depth)
            K_prime = K * (hp/Rp)**2 # Kanagawa K' parameter (used for gap width)

            depth_p = 1 / (1 + 0.04 * K)  
            width = 0.41 * Rp * K_prime**0.25  
    
            # Gaussian-like gap profile centered on planet
            gap_depth = 1 - (1 - depth_p) * np.exp(-((grid.Rc - Rp) / width)**4)

            # apply gap profile using set_gap_profile method, preturbs the viscosity profile to induce a gap.
            disc.set_gap_profile(gap_depth)

        elif gap_params['type'] == 'suriano2018':
            '''
            Wind driven ring: Inserts a pressure bump that represents a wind driven ring
            using a Gaussian profile with a specified width and depth.

            create rings between 1 and 1000 AU with the width getting larger with radius and the depth getting smaller with radius.
            The rings are not centered on a planet, but rather represent a generic wind driven ring.
            '''

            # before 5 AU. Inner bumps are much larger than those after 5AU
            substructures = []
            ring_width = 0.1
            ring_depth = 0.09
            ring_loc = 1.5
            gap_width = 0.1
            gap_depth = 0.3 
            for x in range(7):
                # wind ring
                ring_height = 1 / (1 - (1 - ring_depth) * np.exp(-((grid.Rc - ring_loc) / ring_width)**4))
                # gap introduced by the ring
                gap_height = 1 - (1 - gap_depth) * np.exp(-((grid.Rc - (ring_loc - ring_width - gap_width)) / gap_width)**4)

                # for small transition bump
                if x == 7:
                    ring_height = 1 / (1 - (1 - ring_depth+0.05) * np.exp(-((grid.Rc - ring_loc) / ring_width)**4))

                substructures.append(ring_height)
                substructures.append(gap_height)

                ring_loc += 0.5

            # for after 5 AU
            ring_width = 0.25
            ring_depth = 0.15
            gap_width = 0.3
            gap_depth = 0.0005
            ring_loc = 5.5

            multiplier = 0

            for _ in range(5):

                # wind ring
                ring_height = 1 / (1 - (1 - ring_depth) * np.exp(-((grid.Rc - ring_loc) / ring_width)**4))

                # gap created by the bump
                gap_height = 1 - (1 - gap_depth) * np.exp(-((grid.Rc - (ring_loc - ring_width - gap_width)) / gap_width)**4)

                avg = (ring_height + gap_height) / 2
                substructures.append(avg)

                ring_width += 0.05
                ring_depth += 0.05
                ring_loc += 1.5 + multiplier
                gap_width += 0.1

                multiplier += 0.5

            substructures = np.array(substructures)

            # Treat baseline values (~1) as no features
            mask = ~np.isclose(substructures, 1, atol=1e-12)

            # Mean only over non-1 values at each radius
            num = np.where(mask, substructures, 0).sum(axis=0)
            den = mask.sum(axis=0)

            profile = np.where(den > 0, num / den, 1) # if there are no features at a radius, keep it as 1
            disc.set_gap_profile(profile)


    
    def mass_flux(disc):
        """Compute the radial flux of dust in the disk.
        
        Parameters
        ----------
        disc : Disc
            The disc object containing the dust properties.

        Returns
        -------
        flux : ndarray
            The radial flux of dust and pebbles at each grid point.
        """

        '''
        v_gas = disc._gas.viscous_velocity(disc, disc.Sigma)

        # for dust grains
        dust_v = (disc.v_drift[0][:-1] + (v_gas * disc.Sigma_G[:-1] / (disc.Sigma_D.sum(0)[:-1] + disc.Sigma_G[:-1]))) / (1 - disc_params['d2g'])
        dust_v_cm = dust_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s
        r_cm = grid.Rc[:-1] * AU # convert from AU to cm

        f_D = 2 * np.pi * r_cm * disc.Sigma_D[0][:-1] * dust_v_cm # g/s

        # for pebbles
        pebble_v = (disc.v_drift[1][:-1] + (v_gas * disc.Sigma_G[:-1] / (disc.Sigma_D.sum(0)[:-1] + disc.Sigma_G[:-1]))) / (1 - disc_params['d2g'])
        pebble_v_cm = pebble_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s
        r_cm = grid.Rc[:-1] * AU # convert from AU to cm

        f_P = 2 * np.pi * r_cm * disc.Sigma_D[1][:-1] * pebble_v_cm # g/s
        '''


        v_gas = disc._gas.viscous_velocity(disc, disc.Sigma)
        fg = disc.Sigma_G / (disc.Sigma_D.sum(0) + disc.Sigma_G)

        # for dust grains
        dust_v = v_gas + disc.v_drift[0][1:] / fg[1:]
        dust_v_cm = dust_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s
        r_cm = grid.Rc[1:] * AU # convert from AU to cm

        f_D = 2 * np.pi * r_cm * disc.Sigma_D[0][1:] * dust_v_cm # g/s

        # for pebbles
        pebble_v = v_gas + disc.v_drift[1][1:] / fg[1:]
        pebble_v_cm = pebble_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s
        r_cm = grid.Rc[1:] * AU # convert from AU to cm

        f_P = 2 * np.pi * r_cm * disc.Sigma_D[1][1:] * pebble_v_cm # g/s

        return f_D, f_P, dust_v_cm, pebble_v_cm


    # find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(vr[0])

    # Run model
    # ========================
    t = 0
    n = 0
    data = {}
    data['parameters'] = {
        "Mdot": Mdot,
        "alpha": disc._eos._alpha_t,
        'psi': wind_params['psi_DW'],
        "Mdisk": disc.Mtot()/Msun,
        "Rd": disc.RC(),
        "Mtot": disc.Mtot()/Msun}
    data["R"] = grid.Rc.copy().tolist()
    data["Sigma_G"] = []
    data["Sigma_dust"] = []
    data["Sigma_pebbles"] = []
    data['pebble_size'] = []
    data['gas_velocity'] = []
    data['pebble_drift_velocity'] = []
    data['dust_drift_velocity'] = []
    data['pebble_velocity'] = []
    data['dust_velocity'] = []
    data['time'] = sim_params['t_interval']
    data['Mdot'] = []
    data['Mtot'] =[]
    data['Pressure'] = []
    data['viscosity'] = []
    data['pressure_gradient'] = []
    data['pebble_flux'] = []
    data['dust_flux'] = []
    data['temperature'] = []
    data['stokes_number'] = []
    if gap_params['on'] and (gap_params['type'] == 'duffell2019' or gap_params['type'] == 'kanagawa2016'):
        data['gap_profile'] = {
            'Mp': planet_params['Mp'][0],
            'Rp': planet_params['Rp'][0],
            'inner_edge': float(edge()[0]),
            'inner_edge_idx': int(edge()[2]),
            'outer_edge': float(edge()[1]),
            'outer_edge_idx': int(edge()[3])}
    
    # Define physical bounds for grain sizes
    a_min = 1e-10  # cm (sub-micron floor)
    a_max = 1e10   # cm (meter-scale ceiling)
    
    def fix_stokes(disc, gas):
        """Scale the pebble size to fix the Stokes number across the disk

        Parameters
        ----------
        disc : Disc
            The disc object containing the dust properties.
        gas : ndarray
            The gas surface density at current time.

        Returns
        -------
        a_dust : ndarray
            The scaled dust grain size at each grid point.
        a_pebble : ndarray
            The scaled pebble size at each grid point.
        """
        # Protect against division by zero with floor on reference density
        sigma_ratio = gas / np.maximum(sigma_g0, 1e-30)
        
        # Scale grain sizes and clip to physical bounds
        a_dust = np.clip(a_0_dust * sigma_ratio, a_min, a_max)
        a_pebble = np.clip(a_0_pebble * sigma_ratio, a_min, a_max)

        return a_dust, a_pebble


    if alpha_SS > 5e-3:
        print ("Not Running model - alpha too high.  Alpha, Rd, Mdisk=",eos.alpha, Rd, disc.Mtot()/Msun)
    else:   
        print ("Running model.  Alpha, Rd, Mdisk=",eos.alpha, Rd, disc.Mtot()/Msun)
        for ti in times:
            while t < ti:

                # find timestep given gas and dust maximum timesteps
                dt = ti - t
                if transport_params['gas_transport']:
                    dt = min(dt, disc._gas.max_timestep(disc))
                if transport_params['radial_drift']:
                    dt = min(dt, dust.max_timestep(disc))

                # Extract updated dust frac to update gas
                dust_frac = None
                try:
                    dust_frac = disc.dust_frac
                except AttributeError:
                    pass

                # Extract gas tracers
                gas_chem, ice_chem = None, None
                try:
                    gas_chem = disc.chem.gas.data
                    ice_chem = disc.chem.ice.data
                except AttributeError:
                    pass 

                # Do gas evolution
                if transport_params['gas_transport']:
                    disc._gas(dt, disc, [dust_frac, gas_chem, ice_chem])

                # Do dust evolution
                if transport_params['radial_drift']:
                    dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

                if diffuse is not None:
                    if dust_frac is not None:
                        dust_frac[:] += dt * diffuse(disc, dust_frac[:])

                # Pin the values to >= 0 and <=1:
                disc.Sigma[:] = np.maximum(disc.Sigma, 0)     
                disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
                disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)

                # update disc
                disc.update(dt)

                # increase time and go forward a step
                t += dt
                n += 1

                # print status every 1000 steps
                if (n % 1000) == 0:
                    print('\rNstep: {}'.format(n), flush="True")
                    print('\rTime: {} Myr'.format(t / (1.e6* 2 * np.pi)), flush="True")
                    print('\rdt: {} yr'.format(dt / (2 * np.pi)), flush="True")
            
            # used for calculating Mdot at each time step
            vr = disc._gas.viscous_velocity(disc, disc.Sigma_G)

            dust_flux, pebble_flux, dust_v_cm, pebble_v_cm = mass_flux(disc)

            # appending data for output 
            data["Sigma_G"].append(disc.Sigma_G.copy().tolist())
            data["Sigma_dust"].append(disc.Sigma_D[0].copy().tolist())
            data["Sigma_pebbles"].append(disc.Sigma_D[1].copy().tolist())
            data['pebble_size'].append(disc.grain_size[1].copy().tolist())
            data['gas_velocity'].append(vr.copy().tolist())
            data['pebble_drift_velocity'].append(disc.v_drift[1].copy().tolist())
            data['dust_drift_velocity'].append(disc.v_drift[0].copy().tolist())
            data['pebble_velocity'].append(pebble_v_cm.copy().tolist())
            data['dust_velocity'].append(dust_v_cm.copy().tolist())
            data['Mdot'].append(disc.Mdot(vr[0]))
            data['Mtot'].append(disc.Mtot()/Msun)
            data['Pressure'].append(disc.P.copy().tolist())
            data['viscosity'].append(disc.nu.copy().tolist())
            data['pressure_gradient'].append(gammaP(disc).copy().tolist())
            data['pebble_flux'].append(pebble_flux.copy().tolist())
            data['dust_flux'].append(dust_flux.copy().tolist())
            data['stokes_number'].append(disc.Stokes().copy().tolist())
            data['temperature'].append(disc.T.copy().tolist())


        if not wind_params["on"]:
            wind_params["psi_DW"] = 0

        # Save data to json
        with open(f"denzell_scripts/Data_Updated/flux_vs_alpha/test_nogap_alpha={disc_params['alpha']:.1e}.json", "w") as f:
            json.dump(data, f)

#planetgap_Mp={planet_params['Mp'][0]}_alpha={disc_params['alpha']:.1e}.json
if __name__ == "__main__":
    # Define configuration dictionary
    config = {
        "grid": {
            "rmin": 0.1,
            "rmax": 1000,
            "nr": 1000,
            "spacing": "natural",
            "smart_bining": False,
            "type": "Booth-Mdot" # "LBP", "Booth-alpha", "Booth-Rd", "Booth-Mdot", "winds-alpha", "winds-Rd", "winds-Mdot" or "tabone"
        },
        "star": {
            "M": 1.0, # Solar masses
            "R": 2.5, # Solar radii
            "T_eff": 4000 # Kelvin
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1,
            "t_interval": [0,0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2], #Myr
        },
        "disc": {
            "alpha": 1e-4,
            "M": 0.05, # solar masses
            "d2g": 0.01,
            "Mdot": 1e-8, # solar masses per year
            "Sc": 1.0, # schmidt number
            "Rd": 50, # AU
            'gamma': 1
        },
        "eos": {
            "type": "LocallyIsothermalEOS", # "SimpleDiscEOS", "LocallyIsothermalEOS", or "IrradiatedEOS"
            "opacity": "Tazzari",
            "h0": 0.05455,
            "q": -1/4,
            "Tmax": 1500.
        },
        "transport": {
            "gas_transport": True,
            "radial_drift": True,
            "diffusion": True,
            "van_leer": False
        },
        "dust_growth": {
            "feedback": True,
            "settling": True,
            "f_ice": 1,
            "uf_0": 500,          # Fragmentation velocity for ice-free grains (cm/s)
            "uf_ice": 500,       # Set same as uf_0 to ignore ice effects
            "thresh": 0.5        # Set high threshold to prevent ice effects
        },
        "chemistry": {
            "on"   : False, 
            "fix_mu" : False,
            "mu"     : 2.5,
            "chem_model": "Equilibrium",
            "assert_d2g": False
        },
        "planets": {
            'include_planets': False,
            "planet_model": "Bitsch2015Model",
            "Rp": [10], #[1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [1], 
            "implant_time": [2], # 2pi*t(years)
            "pb_gas_f": 0.05, # Percent of accreted solids converted to gas
            "migrate" : False,
            "pebble_accretion": False,
            "gas_accretion": False, 
            "planetesimal_accretion": False
        },
        "planetesimal": {
            "active": False,
            "diameter": 200,
            "St_min": 1e-2,
            "St_max": 10,
            "pla_eff": 0.05
        },
        "winds": {
            "on": False,
            "psi_DW": 10,
            "e_rad": 0.9
        },
        "gap": {
            'on':False,
            'type':'duffell2019', # 'duffell2019' or 'kanagawa2016'
        }
    }

    run_model(config)


