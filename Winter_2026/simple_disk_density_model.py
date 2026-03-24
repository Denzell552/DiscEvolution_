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
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.chemistry import *
from copy import deepcopy

import time
start_time = time.time()


def run_model(config):

    """
    Rune the disk evolution model and plot the results
    
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

    # Preparing plots
    # ========================

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    plt.subplots_adjust(bottom=0.6, top=0.9)

    # find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(vr[0])
        
    # display disk characteristics
    plt.figtext(0.5, 0.01, f"Mdot={Mdot:.3e}, alpha={disc._eos._alpha_t:.3e}, Mtot={disc.Mtot()/Msun:.3e}, Rd={disc.RC():.3e}", ha="center", fontsize=12)

    # this is to synchronize colors
    d = 0

    cm2 = plt.get_cmap("viridis")

    # gradient colors also present to give options
    color1=iter(plt.cm.Blues(np.linspace(0.4, 1, 8)))
    color2=iter(plt.cm.Greys(np.linspace(0.4, 1, 8)))
    color3=iter(plt.cm.Greens(np.linspace(0.4, 1, 8)))
    color4=iter(plt.cm.Reds(np.linspace(0.4, 1, 8)))

    # Run model
    # ========================
    t = 0
    n = 0
    data = {}
    data["R"] = []
    data["Sigma_G"] = []
    data["Sigma_dust"] = []
    data["Sigma_pebbles"] = []
    #data["Sigma_planetesimals"] = []
    data["T"] = []

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
                
        

            # appending data for output
            data["R"].append(grid.Rc.copy().tolist())
            data["Sigma_G"].append(disc.Sigma_G.copy().tolist())
            data["Sigma_dust"].append(disc.Sigma_D[0].copy().tolist())
            data["Sigma_pebbles"].append(disc.Sigma_D[1].copy().tolist())
            data["T"].append(disc.T.tolist())


            # plot evolution
            axes[0].plot(grid.Rc, disc.Sigma_G, color=next(color1), label='{:.3f} Myr'.format(t / (1.e6 * 2 * np.pi)))
            axes[1].plot(grid.Rc, disc.Sigma_D[0], color=next(color2), label='{:.3f} Myr'.format(t / (1.e6 * 2 * np.pi)))
            axes[2].plot(grid.Rc, disc.Sigma_D[1], color=next(color3), label='{:.3f} Myr'.format(t / (1.e6 * 2 * np.pi)))


        '''         
        plot gas surface density, charateristic dust size and drift timescale vs radius 
        '''

        if not wind_params["on"]:
            wind_params["psi_DW"] = 0

        # plotting gas surface density over radius
        #axes[0].plot(grid.Rc, disc.Sigma_G, color=c1, label='Gas Surface Density')
        axes[0].set_xscale('log')        
        axes[0].set_yscale('log')
        axes[0].minorticks_off()
        axes[0].legend(fontsize=12)
        axes[0].set_xlabel('Radius (AU)', fontsize=15)
        axes[0].set_ylabel('Surface Density ($g/cm^2$)', fontsize=15)
        axes[0].set_title('Gas Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)
        
        # plotting dust surface density over radius
        #axes[1].plot(grid.Rc, disc.Sigma_D[0], color=c2, label='Dust Surface Density')
        axes[1].set_xscale('log')        
        axes[1].set_yscale('log')
        axes[1].set_ylim(10**-7, 10**2)
        axes[1].minorticks_off()
        axes[1].legend(fontsize=12)
        axes[1].set_xlabel('Radius (AU)', fontsize=15)
        axes[1].set_ylabel('Surface Density ($g/cm^2$)', fontsize=15)
        axes[1].set_title('Dust Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)

        # plotting pebble surface density over radius
        # axes[2].plot(grid.Rc, disc.Sigma_D[1], color=c3, label='Pebble Surface Density')
        axes[2].set_xscale('log')        
        axes[2].set_yscale('log')
        axes[2].set_ylim(10**-7, 10**3)
        axes[2].minorticks_off()
        axes[2].legend(fontsize=12)
        axes[2].set_xlabel('Radius (AU)', fontsize=15)
        axes[2].set_ylabel('Surface Density ($g/cm^2$)', fontsize=15)
        axes[2].set_title('Pebble Surface Density with Ψ = {}'.format(wind_params["psi_DW"]), fontsize=17)


        plt.tight_layout(pad=3.5)
        
        # saving figure
        fig.savefig(f"Winter_2026/Figs/model_test_winds_mig_psi={wind_params['psi_DW']}_alpha={disc_params['alpha']:.1e}_M={disc_params['M']:.1e}_Rd={disc_params['Rd']:.1e}.png")

        # Save data to json
        with open(f"Winter_2026/Data/model_test_winds_mig_psi={wind_params['psi_DW']}_alpha={disc_params['alpha']:.1e}_M={disc_params['M']:.1e}_Rd={disc_params['Rd']:.1e}.json", "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    # Define configuration dictionary
    config = {
        "grid": {
            "rmin": 1,
            "rmax": 1000,
            "nr": 1000,
            "spacing": "natural",
            "smart_bining": False,
            "type": "Booth-Mdot" # "LBP", "Booth-alpha", "Booth-Rd", "winds-alpha", or "Booth-Mdot"
        },
        "star": {
            "M": 1.0, # Solar masses
            "R": 2.5, # Solar radii
            "T_eff": 4000 # Kelvin
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1,
            "t_interval": [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], #Myr
        },
        "disc": {
            "alpha": 1e-3,
            "M": 0.05, # solar masses
            "d2g": 0.01,
            "Mdot": 8.85e-9, # for Tmax=1500
            "Sc": 1.0, # schmidt number
            "Rd": 50, # AU
            'gamma': 1
        },
        "eos": {
            "type": "IrradiatedEOS", # "SimpleDiscEOS", "LocallyIsothermalEOS", or "IrradiatedEOS"
            "opacity": "Tazzari",
            "h0": 0.025,
            "q": -0.2,
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
            "Rp": [], #[1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [], #[0.1, 0.1, 0.1, 0.1, 0.1], # initial mass of embryo [M_Earth]
            "implant_time": [], # 2pi*t(years)
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
        }
    }

    run_model(config)


