import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('/Users/denzell/Documents/W26_Coop/DiscEvolution_/') 

import numpy as np
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

    elif grid_params['type'] == 'power_law':
        # For power law grid, just create a power law Sigma profile with desired Mdisk
    
        # extract parameters
        R = grid.Rc
        p = disc_params['p']
        Mdot = disc_params['Mdot'] 
        alpha = disc_params['alpha']

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])

        # Initialize EOS with a guess Sigma to get sound speed
        Sigma_guess = np.ones_like(R) * 1e3  # Initial guess
        eos.set_grid(grid)
        eos.update(0, Sigma_guess)
        
        # define Sigma profile using sound speed from EOS
        Sigma0 = star.Omega_k(R[0]) / (3*np.pi*alpha*eos.cs[0]**2) * Mdot / (2*np.pi*AU**2) * Msun  
        Sigma = Sigma0 * (R/R[0])**(p)

        gas_temp = ViscousEvolution()

        # iterate for correct Mdot
        for j in range(100):
            # define a temporary disc to compute Mdot
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, Sigma=Sigma)) 

            # Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0] 
            Sigma = 0.5 * (Sigma + Sigma_new) 

            # update the eos with actual Sigma values
            eos.update(0, Sigma)
        
        alpha_SS = alpha

    # Set up dynamics
    # ========================
    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW)
        else:
            gas = ViscousEvolution()
    
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
            ''' Gap profile 1: Using current gap profile from planet.py (Duffell 2019) '''
        
            planet = Planet(
                Mp=planet_params['Mp'][0],
                ap=planet_params['Rp'][0],
            )

            Mp = planet_params['Mp'][0]
            Rp = planet_params['Rp'][0]

            # gap depth array
            gap_depth = planet.gap_profile(disc)

            # apply gap to disc surface density
            #disc.Sigma[:] *= depth
            #disc.update(0)

            # OR apply gap profile using set_gap_profile method
            disc.set_gap_profile(gap_depth)

            def edge():
                '''Calculate the location of the gap edge, defined as where the gap starts to dip'''
                
                edges = []

                for i in gap_depth:
                    if i / np.max(gap_depth) < 0.98: # if gap depth is 98% of max depth, we are at the edge
                        idx = np.argmin(np.abs(gap_depth - i))
                        edges.append(grid.Rc[idx])

                idx_inner = np.argmin(np.abs(grid.Rc - edges[0]))
                idx_outer = np.argmin(np.abs(grid.Rc - edges[-1]))
    
                return edges[0], edges[-1], idx_inner, idx_outer
    
        elif gap_params['type'] == 'kanagawa2016':
            ''' Gap profile 2: Using kanagawa 2016 to set gap profile '''

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


            # apply gap to disc surface density
            #disc.Sigma[:] *= gap_depth
            #disc.update(0)
    
            # OR apply gap profile using set_gap_profile method
            disc.set_gap_profile(gap_depth)
   
    
    def M_flux(disc):
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
        global dust_v_cm, pebble_v_cm

        v_gas = disc._gas.viscous_velocity(disc, disc.Sigma)

        # for dust grains
        dust_v = (disc.v_drift[0][:-1] + (v_gas * disc.Sigma_G[:-1] / (disc.Sigma_D.sum(0)[:-1] + disc.Sigma_G[:-1]))) / (1 - disc_params['d2g'])
        f_D = 2 * np.pi * grid.Rc[:-1] * disc.Sigma_D[0][:-1] * np.abs(dust_v)
        flux_D = f_D * AU**2 / Mearth * yr # convert from g/s to Earth masses per year

        dust_v_cm = dust_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s

        # for pebbles
        pebble_v = (disc.v_drift[1][:-1] + (v_gas * disc.Sigma_G[:-1] / (disc.Sigma_D.sum(0)[:-1] + disc.Sigma_G[:-1]))) / (1 - disc_params['d2g'])
        f_P = 2 * np.pi * grid.Rc[:-1] * disc.Sigma_D[1][:-1] * np.abs(pebble_v)
        flux_P = f_P * AU**2 / Mearth * yr # convert from g/s to Earth masses per year

        pebble_v_cm = pebble_v * AU * yr / 3.15e7 # convert from AU/yr to cm/s

        return flux_D, flux_P


    def compute_Mdot(disc):
        """Compute the mass accretion rate of dust and pebbles in the disk.
        
        Parameters
        ----------
        disc : Disc
            The disc object containing the dust properties.

        Returns
        -------
        Mdot: ndarray
            The mass accretion rate at each grid point.
        """

        Mdot = 3 * np.pi * alpha_SS * eos.cs**2 * disc.Sigma_G / star.Omega_k(grid.Rc) * (2*np.pi*AU**2/Msun)# in Msun/yr
        return Mdot

    # Preparing plots
    # ========================
    Mp = planet_params['Mp'][0]
    Rp = planet_params['Rp'][0]

    fig, axes = plt.subplots(6, 2, figsize=(20, 36))
    plt.subplots_adjust(bottom=0.6, top=0.9)

    # find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, disc.Sigma)
    Mdot = disc.Mdot(vr[0])
        
    # display disk characteristics
    plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}, alpha={disc._eos._alpha_t:.3e}, Mtot={disc.Mtot()/Msun:.3e}", ha="center", fontsize=12)
    fig.suptitle(f'Disc Evoltion with Planetary Gap q = {Mp*Mjup / (star_params["M"]*Msun):.2e}', fontsize=20)


    # gradient colors also present to give options
    color1=iter(plt.cm.Blues(np.linspace(0.4, 1, 9)))
    color2=iter(plt.cm.Greys(np.linspace(0.4, 1, 9)))
    color3=iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))
    color4=iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))
    color5=iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
    color6=iter(plt.cm.Purples(np.linspace(0.4, 1, 9)))
    color7=iter(plt.cm.Greens(np.linspace(0.4, 1, 6)))
    color8=iter(plt.cm.Reds(np.linspace(0.4, 1, 9)))
    color9=iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
    color10=iter(plt.cm.Oranges(np.linspace(0.4, 1, 9)))
    color11=iter(plt.cm.Greens(np.linspace(0.4, 1, 9)))

    # Run model
    # ========================
    t = 0
    n = 0
    data = {}
    data["R"] = []
    data['R/Rp'] = []
    data["Sigma_G"] = []
    data["Sigma_dust"] = []
    data["Sigma_pebbles"] = []
    data['pebble_size'] = []
    data['Mdot'] = []
    data['Mdot_calc0'] = []
    data['Mdot_calc1'] = []
    data['dust_flux_fraction'] = []
    data['dust_flux'] = []
    data['dust_velocity'] = []
    data['orbits'] = []

    # fix grain size and initialize dust profile. index 79 comes from where r=1.5=7.8AU
    disc.grain_size[0] = np.ones_like(disc.grain_size[0]) * 1 # cm
    disc.dust_frac[0][79:] = np.ones_like(disc.dust_frac[0][79:]) * 0.01 # outside of the gap
    disc.dust_frac[0][:79] = np.ones_like(disc.dust_frac[0][:79]) * 1e-20 # inside and before the gap
    disc.dust_frac[1] = np.ones_like(disc.dust_frac[1]) * 0
    
    # do dust evolution for first time step to initialize drift velocity and fluxes before starting time loop, otherwise get numerical issues with zero drift velocity at t=0 causing dust to not evolve at all in first step and then having a large jump in dust properties in second step.
    dust(0, disc, gas_tracers=None, dust_tracers=None)
    disc._gas(0, disc, [disc.dust_frac, None, None])
    disc.update(0)

    if alpha_SS > 5e-3:
        print ("Not Running model - alpha too high.  Alpha, Rd, Mdisk=",eos.alpha, disc.Mtot()/Msun)
    else:   
        print ("Running model.  Alpha, Rd, Mdisk=",eos.alpha, disc.Mtot()/Msun)
        for ti in times:
            while t < ti:
                
                if t / (1.e6 * 2 * np.pi) < 0.11859672847741:
                    disc.dust_frac[0][79:] = np.ones_like(disc.dust_frac[0][79:]) * 0.01 # outside of the gap
                    disc.dust_frac[0][:79] = np.ones_like(disc.dust_frac[0][:79]) * 1e-20 # inside and before the gap
                
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
                    
                    # to preserve planetesimal surface density so 
                    # that it doesn't move with a change in Sigma 
                    # as a whole, we do the following:
                    if disc._planetesimal:
                        disc._gas(dt, disc, [dust_frac[:-1], gas_chem, ice_chem])
                    else:
                        disc._gas(dt, disc, [dust_frac, gas_chem, ice_chem])

                # Do dust evolution
                if transport_params['radial_drift']:
                    dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

                if diffuse is not None:

                    if gas_chem is not None:
                        gas_chem[:] += dt * diffuse(disc, gas_chem)
                    if ice_chem is not None:
                        ice_chem[:] += dt * diffuse(disc, ice_chem) #### may use planetesimals to move, double check

                    if dust_frac is not None:
                        if disc._planetesimal:
                            # excluding planetesimals (assume they don't move)
                            dust_frac[:2] += dt * diffuse(disc, dust_frac[:2]) 
                        else: 
                            dust_frac[:] += dt * diffuse(disc, dust_frac[:])

                # Pin the values to >= 0 and <=1:
                disc.Sigma[:] = np.maximum(disc.Sigma, 0)     
                disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
                disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
                
                if t / (1.e6 * 2 * np.pi) < 0.11859672847741:
                    disc.dust_frac[0][79:] = np.ones_like(disc.dust_frac[0][79:]) * 0.01 # outside of the gap
                    disc.dust_frac[0][:79] = np.ones_like(disc.dust_frac[0][:79]) * 1e-20 # inside and before the gap
                
                # update disc
                disc.update(dt)

                # increase time and go forward a step
                t += dt
                n += 1

                # Orbit counter
                R_orb = planet_params['Rp'][0]
                Omega = star.Omega_k(R_orb)  
                n_orbits = Omega * t / (2 * np.pi)

                # print status every 1000 steps
                if (n % 1000) == 0:
                    print(f"Orbit number at {R_orb:.2f} AU: {n_orbits:.2f}", flush='True')
                    print('\rNstep: {}'.format(n), flush="True")
                    print('\rTime: {} Myr'.format(t / (1.e6* 2 * np.pi)), flush="True")
                    print('\rdt: {} yr'.format(dt / (2 * np.pi)), flush="True")
            
            # for calculating Mdot at current time step
            vr = disc._gas.viscous_velocity(disc, disc.Sigma) 

            '''
            # calculating dust flux at gap edge
            half_depth = (np.max(gap_depth) + np.min(gap_depth)) / 2
            inner_edge = np.argmin(np.abs(gap_depth[:np.argmin(np.abs(grid.Rc - Rp))] - half_depth)) # calculates inner edge at half max depth
            outer_edge = np.argmin(np.abs(gap_depth[np.argmin(np.abs(grid.Rc - Rp)):] - half_depth)) + np.argmin(np.abs(grid.Rc - Rp)) # calculates inner edge at half max depth   
            '''

            flux_inner = M_flux(disc)[0][edge()[2]]
            flux_outer = M_flux(disc)[0][edge()[3]]

            flux_fraction = flux_inner / flux_outer

            #total flux in Mearth/yr
            flux = M_flux(disc)[0]

            # appending data for output 
            data["R"].append(grid.Rc.copy().tolist())
            data["Sigma_G"].append(disc.Sigma_G.copy().tolist())
            data["Sigma_dust"].append(disc.Sigma_D[0].copy().tolist())
            data["Sigma_pebbles"].append(disc.Sigma_D[1].copy().tolist())
            data['pebble_size'].append(disc.grain_size[0].copy().tolist())
            data["R/Rp"].append((grid.Rc / Rp).copy().tolist())
            data['Mdot'].append(float(disc.Mdot(vr[0])))
            data['Mdot_calc0'].append(compute_Mdot(disc)[0].tolist())
            data['Mdot_calc1'].append(compute_Mdot(disc)[1].tolist())
            data['dust_flux_fraction'].append(flux_fraction.tolist())
            data['dust_flux'].append(flux.tolist())
            data['dust_velocity'].append(dust_v_cm.copy().tolist())
            try:
                data['orbits'].append(n_orbits)
            except:
                data['orbits'].append(0)

            # for comparison of nu/r vs r. Should be proportional to r^-1/2. Vr vs r should be proportional to -3/2
            nu_r = eos.nu / grid.Rc
            nu_sigma = (eos.nu * AU**2 * 2*np.pi / 3.15e7) * disc.Sigma_G
            cs2_sigma = eos.cs**2 * disc.Sigma_G

            vr_cm = vr * AU * 2*np.pi / 3.15e7 
            nu_r_cm = nu_r * AU * 2*np.pi / 3.15e7
            
            # plotting gas surface density, grain size and pebble density
            try:
                axes[0][0].loglog(grid.Rc, disc.Sigma_G, color=next(color1), label=f'{n_orbits:.2f} orbits')
                axes[0][1].semilogx(grid.Rc, disc.grain_size[0], color=next(color2), label=f'{n_orbits:.2f} orbits')
                axes[1][0].loglog(grid.Rc, disc.Sigma_D[0], color=next(color3), label=f'{n_orbits:.2f} orbits')
                axes[1][1].loglog(grid.Rc, disc.Stokes()[0], color=next(color4), label=f'{n_orbits:.2f} orbits') 
                axes[2][0].plot(grid.Rc, disc.v_drift[0], color=next(color5), label=f'{n_orbits:.2f} orbits') 
                if t / (1.e6 * 2 * np.pi) > 0.11859672847741*1.25:
                    axes[4][0].loglog(grid.Rc[:-1], flux, color=next(color7), label=f'{n_orbits:.2f} orbits')
                axes[3][0].loglog(grid.Rc[:-1], dust_v_cm, color=next(color6), label=f'{n_orbits:.2f} orbits')
                axes[5][0].semilogx(grid.Rc, nu_sigma, color=next(color9), label=f'{n_orbits:.2f} orbits')
                axes[5][1].semilogx(grid.Rc[:-1], np.abs(vr_cm), color=next(color10), label=f'{n_orbits:.2f} orbits') 
                axes[5][1].semilogx(grid.Rc, np.abs(-3/2 * nu_r_cm), color=next(color11))
            
            except:
                axes[0][0].loglog(grid.Rc, disc.Sigma_G, color=next(color1), label=f'0 orbits')
                axes[0][1].semilogx(grid.Rc, disc.grain_size[0], color=next(color2), label=f'0 orbits')
                axes[1][0].loglog(grid.Rc, disc.Sigma_D[0], color=next(color3), label=f'0 orbits')
                axes[1][1].loglog(grid.Rc, disc.Stokes()[0], color=next(color4), label=f'0 orbits') 
                axes[2][0].plot(grid.Rc, disc.v_drift[0], color=next(color5), label=f'0 orbits') 
                axes[3][0].loglog(grid.Rc[:-1], dust_v_cm, color=next(color6), label=f'0 orbits')
                #axes[4][0].loglog(grid.Rc[:-1], flux, color=next(color7), label=f'0 orbits')
                axes[5][0].semilogx(grid.Rc, nu_sigma, color=next(color9), label=f'0 orbits')
                axes[5][1].semilogx(grid.Rc[:-1], np.abs(vr_cm), color=next(color10), label=f'0 orbits') 
                axes[5][1].semilogx(grid.Rc, np.abs(-3/2 * nu_r_cm), color=next(color11))


        if not wind_params["on"]:
            wind_params["psi_DW"] = 0

        # plotting configuration
        for row in range(len(axes)):
            for column in range(len(axes[row])):
                axes[row][column].minorticks_off()
                axes[row][column].legend(fontsize=10)
                axes[row][column].grid(True)
                axes[row][column].tick_params(axis='both', which='major', labelsize=13)


        axes[0][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[0][0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=15)
        axes[0][0].set_title('Gas Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        #axes[0][0].set_ylim(1e-3, 1e10)
        
        axes[0][1].set_xlabel('Radius (AU)', fontsize=15)
        axes[0][1].set_ylabel('Grain Size (cm)', fontsize=15)
        axes[0][1].set_title('Characteristic Pebble Size with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)

        axes[1][0].axvline(edge()[0], color='red', linestyle='--', label='Inner Gap Edge')
        axes[1][0].axvline(edge()[1], color='green', linestyle='--', label='Outer Gap Edge')
        axes[1][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[1][0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=15)
        axes[1][0].set_title('Dust Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        axes[1][0].set_ylim(1e-15, 1e5)
        
        axes[1][1].set_xlabel('Radius (AU)', fontsize=15)
        axes[1][1].set_ylabel('Stokes Number', fontsize=15)
        axes[1][1].set_title('Stokes Number with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        #axes[1][1].set_ylim(1e-10, 1e13)

        axes[2][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[2][0].set_ylabel('Drift Velocity (cm/s)', fontsize=15)
        axes[2][0].set_title('Pebble Drift Velocity and Pressure Gradient',fontsize=17)
        axes[2][0].set_xscale('log')
        axes[2][0].set_yscale('symlog', linthresh=1e-10)
        #axes[2][0].set_ylim(-1e-5, 1e-3)
        axes20 = axes[2][0].twinx()
        axes20.plot(grid.Rc, gammaP(disc), color='green', linestyle='--')
        axes20.set_ylabel('Pressure Gradient (Green)', fontsize=15)
        axes20.tick_params(axis='y', labelsize=13)

        axes[2][1].plot(data['orbits'], data['Mdot'], marker='o', color='black', label='Mdot DiscEvolution')
        axes[2][1].set_xlabel('Orbit number', fontsize=15)
        axes[2][1].set_ylabel('Mdot (Msun/yr)', fontsize=15)
        axes[2][1].set_title('Mass Accretion Rate with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)

        axes[3][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[3][0].set_ylabel('dust velocty (cm/s)', fontsize=15)
        axes[3][0].set_title('Dust Velocity with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        axes[3][0].set_yscale('symlog')
        axes[3][0].axvline(edge()[0], color='red', linestyle='--', label='Inner Gap Edge')
        axes[3][0].axvline(edge()[1], color='green', linestyle='--', label='Outer Gap Edge')

        axes[3][1].plot(data['orbits'], data['Mdot_calc0'], marker='o', color='green', label='Mdot Calc 0')
        axes[3][1].plot(data['orbits'], data['Mdot_calc1'], marker='o', color='red', label='Mdot Calc 1')
        axes[3][1].set_xlabel('Orbit number', fontsize=15)
        axes[3][1].set_ylabel('Mdot (Msun/yr)', fontsize=15)
        axes[3][1].set_title('Mass Accretion Rate by Calculation with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        axes[3][1].legend(fontsize=10)

        axes[4][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[4][0].set_ylabel('$Flux (M_{earth} / yr)$', fontsize=15)
        axes[4][0].set_title('Mass Flux with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        axes[4][0].axvline(edge()[0], color='red', linestyle='--', label='Inner Gap Edge')
        axes[4][0].axvline(edge()[1], color='green', linestyle='--', label='Outer Gap Edge')

        axes[4][1].plot(data['orbits'], data['dust_flux_fraction'], marker='o', color='red', label='Dust Flux Fraction')
        axes[4][1].set_xlabel('Number of Orbits', fontsize=15)
        axes[4][1].set_ylabel('Dust Flux Fraction', fontsize=15)
        axes[4][1].set_title('Dust Flux Fraction of inner edge vs outer edge with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)

        axes[5][0].set_xlabel('Radius (AU)', fontsize=15)
        axes[5][0].set_ylabel('$\\Sigma \\nu (g/s)$', fontsize=15)
        axes[5][0].set_title('Viscosity times Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)

        axes[5][1].set_xlabel('Radius (AU)', fontsize=15)
        axes[5][1].set_title('Vr (Orange) vs $-3/2 * \\nu/r$ (Green) with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        axes[5][1].set_ylabel('cm/s', fontsize=15)
        axes[5][1].set_yscale('log')

        plt.tight_layout(pad=3.5)
        
        # saving figure
        fig.savefig(f"Winter_2026/Figs/Weber2018rep/rep_size={disc.grain_size[0][0]:.2f}_q={eos_params['q']}_p={disc_params['p']}_Mp={Mp}Mj_alpha={disc_params['alpha']:.1e}_Mdot={disc_params['Mdot']:.1e}.png")

        # Save data to json
        with open(f"Winter_2026/Data/Weber2018rep/rep_size={disc.grain_size[0][0]:.2f}_q={eos_params['q']}_p={disc_params['p']}_Mp={Mp}Mj_alpha={disc_params['alpha']:.1e}_Mdot={disc_params['Mdot']:.1e}.json", "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    # Define configuration dictionary
    config = {
        "grid": {
            "rmin": 0.1,
            "rmax": 1000,
            "nr": 1000,
            "spacing": "natural",
            "smart_bining": False,
            "type": "power_law" # "LBP", "Booth-alpha", "Booth-Rd", "winds-alpha", or "Booth-Mdot"
        },
        "star": {
            "M": 1.0, # Solar masses
            "R": 2.5, # Solar radii
            "T_eff": 4000 # Kelvin
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1.5,
            "t_interval":[0, 0.059298364238704, 0.11859672847741, 0.11859672847741*1.05, 0.11859672847741*1.25, 0.17789509271611,  0.11859672847741*1.75, 0.237193456954817], # Myr
        },
        "disc": {
            "alpha": 3e-3,
            "M": 0.05, # solar masses
            "d2g": 0.01,
            "Mdot": 1e-7, # for Tmax=1500
            "Sc": 1.0, # schmidt number
            "Rd": 50, # AU
            'gamma': 1,
            'p': -0.5 # power law index for power law grid
        },
        "eos": {
            "type": "LocallyIsothermalEOS", # "SimpleDiscEOS", "LocallyIsothermalEOS", or "IrradiatedEOS"
            "opacity": "Tazzari",
            "h0": 0.05,
            "q": -0.5,
            "Tmax": 1500.
        },
        "transport": {
            "gas_transport": True,
            "radial_drift": True,
            "diffusion": True,
            "van_leer": False
        },
        "dust_growth": {
            "feedback": False,
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
            "Rp": [5.2], #[1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [2], #[0.1, 0.1, 0.1, 0.1, 0.1], # initial mass of embryo [M_Earth]
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
            "psi_DW": 0,
            "e_rad": 0.9
        },
        "gap": {
            'on':True,
            'type':'duffell2019', # 'duffell2019' or 'kanagawa2016'
        }
    }

    run_model(config)

# time at orbit 10 000: 0.11954625817064822 Myr
# time at orbit 20 000: 0.23909251634151285 Myr

'''
for gap diagnostic:
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(grid.Rc[26:116]/Rp, gap_depth[26:116], color='Blue')
ax.axvline(grid.Rc[inner_edge]/Rp, color='green', label='inner')
ax.axvline(grid.Rc[outer_edge]/Rp, color='red', label='outer')
ax.legend()
fig.show()
'''

