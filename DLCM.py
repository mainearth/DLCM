# coding:utf-8
# @Time    : 2024/11/11
# @Author  : Huiwen Li

import os
import time
import numpy as np
from osgeo import osr,ogr,gdal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import random
from numba import jit
from pyswarm import pso  
import warnings
warnings.filterwarnings("ignore")

def deal_with_nc(ncfilepath,variable_name,variable_weidu):
	ncfile=ncfilepath.encode('GB2312')  
	f = Dataset(ncfile)
	v = f.variables
	vndim=v[variable_name].ndim
	if vndim==3:
		rasterArray=v[variable_name][variable_weidu,:,:]	
	elif vndim==2:
		rasterArray=v[variable_name][:,:]
	elif vndim==1:
		rasterArray=v[variable_name][:]
	else:
		print('data dimension wrong')
	return rasterArray


def read_all_files_inside_filepath(filepath,filetype,final_file_list):
	pathDir=os.listdir(filepath)
	for each in pathDir:
		newDir=os.path.join(filepath,each)
		if os.path.isfile(newDir):
			if os.path.splitext(newDir)[1]==filetype:
				final_file_list.append(newDir.replace("\\","/"))			
		else:
			read_all_files_inside_filepath(newDir,filetype,final_file_list)

def get_all_type_files(filepath,filetype):
	final_file_list=[]
	read_all_files_inside_filepath(filepath,filetype,final_file_list)
	return final_file_list


def write_rasterfile(output_rasterfile,im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NodataValue):	
	driver=gdal.GetDriverByName("GTiff")
	datasetnew=driver.Create(output_rasterfile,im_width,im_height,im_bands,gdal.GDT_Float32)
	datasetnew.SetGeoTransform(im_geotrans)
	datasetnew.SetProjection(im_proj)
	datasetnew.GetRasterBand(1).SetNoDataValue(float(NodataValue))
	datasetnew.GetRasterBand(1).WriteArray(im_data)
	del datasetnew


def read_rasterfile_setnan(input_rasterfile):
	dataset=gdal.Open(input_rasterfile)
	im_width=dataset.RasterXSize
	im_height=dataset.RasterYSize
	im_bands=dataset.RasterCount
	im_geotrans=dataset.GetGeoTransform()
	im_proj=dataset.GetProjection()
	im_data=dataset.ReadAsArray(0,0,im_width,im_height).astype(float)
	NoDataValue=dataset.GetRasterBand(1).GetNoDataValue()
	im_data[im_data==NoDataValue]=np.nan
	return [im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]

def get_keys(d, value):
	return [k for k,v in d.items() if v == value]

def write_2dim_list_to_txt_file(csv_file,listdata):
	f=open(csv_file,'w')
	for eachi in listdata[:-1]:
		if len(eachi)==1:
			f.write(str(eachi[0])+'\n')
		else:
			for eachj in eachi[:-1]:
				f.write(str(eachj)+',')
			f.write(str(eachi[-1])+'\n')
	for lasti in listdata[-1][:-1]:
		f.write(str(lasti)+',')
	f.write(str(listdata[-1][-1]))
	f.close()

def write_1dim_list_to_txt_file(csv_file,listdata):
	f=open(csv_file,'w')
	for i in listdata[:-1]:
		f.write(str(i)+',')
	f.write(str(listdata[-1]))
	f.close()


def read_2dim_txt(txt_file):
	f=open(txt_file,'r')
	c=f.read()
	c_all=c.split('\n')
	all_data_array=np.empty([len(c_all),len(c_all[0].split(','))])
	for i in range(len(c_all)):
		all_data_array[i]=c_all[i].split(',')
	return all_data_array


def NPP_Ag_Bg_allocation(NPP,fBNPP=0.41,fr20=0.65,fr20_100=0.31):
	#先给个初始化的值
	fBNPP=0.41
	fr20=0.65
	fr20_100=0.31
	Cab_npp=NPP*(1-fBNPP)
	Croot20_npp=NPP*fBNPP*fr20
	Croot20_100_npp=NPP*fBNPP*fr20_100
	return Cab_npp,Croot20_npp,Croot20_100_npp

def calculate_PE(pmax,C_Young,Km):
	if Km+C_Young==0:
		return 0
	else:
		return pmax*C_Young/(Km+C_Young)

def pbias(estimate,observation):
	if observation==0:
		return 0
	else:
		return (estimate-observation)*100.0/observation


def SoilC_Dynamics_4pools(spin_up_year,NPP):
	Cab_npp,Croot20_npp,Croot20_100_npp=NPP_Ag_Bg_allocation(NPP)

	C_Young_top,C_Old_top=0,0	
	C_Young_sub,C_Old_sub=0,0	
	I_ab_npp_Young=0.04
	
	I_root_npp_Young_top=0.03
	I_root_npp_Young_sub=0.02
	
	I_Young_top_down=0.04
	I_Young_sub_down=0.01
	I_Old_top_down=0.03
	I_Old_sub_down=0.02
	k_ab_npp=0.4
	k_Young_top=0.02
	k_Young_sub=0.001
	k_Old_top=0.0006
	k_Old_sub=0.000006
	e_ab_npp_Old=0.4
	e_Young_Old_top=0.2
	e_Young_Old_sub=0.1
	pmax=2.5
	Km=300   
	
	C_Young_top_dynamics=[C_Young_top]
	C_Old_top_dynamics=[C_Old_top]	
	C_Young_Old_top_all=[C_Young_top+C_Old_top]
	
	C_Young_sub_dynamics=[C_Young_sub]
	C_Old_sub_dynamics=[C_Old_sub]	
	C_Young_Old_sub_all=[C_Young_sub+C_Old_sub]	

	for i in range(spin_up_year):
		C_Young_top_rate=I_ab_npp_Young*Cab_npp+Croot20_npp*I_root_npp_Young_top-I_Young_top_down*C_Young_top-(1-I_Young_top_down)*C_Young_top*k_Young_top
		C_Old_top_rate=Cab_npp*(1-I_ab_npp_Young)*e_ab_npp_Old*k_ab_npp+(1-I_Young_top_down)*C_Young_top*e_Young_Old_top*k_Young_top-I_Old_top_down*C_Old_top-(1-I_Old_top_down)*C_Old_top*k_Old_top*(1+calculate_PE(pmax,C_Young_top,Km))   
		C_Young_sub_rate=I_Young_top_down*C_Young_top+Croot20_100_npp*I_root_npp_Young_sub-I_Young_sub_down*C_Young_sub-(1-I_Young_sub_down)*C_Young_sub*k_Young_sub
		C_Old_sub_rate=I_Old_top_down*C_Old_top+(1-I_Young_sub_down)*C_Young_sub*e_Young_Old_sub*k_Young_sub-I_Old_sub_down*C_Old_sub-(1-I_Old_sub_down)*C_Old_sub*k_Old_sub*(1+calculate_PE(pmax,C_Young_sub,Km)) 
		C_Young_top+=C_Young_top_rate
		C_Old_top+=C_Old_top_rate
		C_Young_sub+=C_Young_sub_rate
		C_Old_sub+=C_Old_sub_rate	
		C_Young_top_dynamics.append(C_Young_top)
		C_Old_top_dynamics.append(C_Old_top)
		C_Young_Old_top_all.append(C_Young_top+C_Old_top)
		C_Young_sub_dynamics.append(C_Young_sub)
		C_Old_sub_dynamics.append(C_Old_sub)
		C_Young_Old_sub_all.append(C_Young_sub+C_Old_sub)		
	return C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all

def get_para_range():
	para_range={}
	para_range['I_ab_npp_Young']=[0,0.05]
	para_range['I_root_npp_Young_top']=[0,0.05]
	para_range['I_root_npp_Young_sub']=[0,0.05]
	para_range['I_Young_top_down']=[0,0.05]
	para_range['I_Young_sub_down']=[0,0.05]
	para_range['I_Old_top_down']=[0,0.05]
	para_range['I_Old_sub_down']=[0,0.05]
	para_range['k_ab_npp']=[0.1,1]
	para_range['k_Young_top']=[0.01,1]
	para_range['k_Young_sub']=[0.01,1]
	para_range['k_Old_top']=[0.000001,0.1]
	para_range['k_Old_sub']=[0.000001,0.1]
	para_range['e_ab_npp_Old']=[0.2,0.8]
	para_range['e_Young_Old_top']=[0.2,0.8]
	para_range['e_Young_Old_sub']=[0.2,0.8]
	para_range['pmax']=[-0.5,4]
	para_range['Km']=[0,500]
	return para_range

def initialize_paratmeters():
	I_ab_npp_Young=0.04 
	I_root_npp_Young_top=0.03
	I_root_npp_Young_sub=0.02
	I_Young_top_down=0.04
	I_Young_sub_down=0.02
	k_Young_top=0.02
	k_Young_sub=0.001
	e_Young_Old_top=0.2
	e_Young_Old_sub=0.1
	pmax=2.5
	Km=300  
	I_Old_top_down=0.03	
	I_Old_sub_down=0.02		
	k_ab_npp=0.4
	k_Old_top=0.0006	
	k_Old_sub=0.000006
	e_ab_npp_Old=0.4 
	parameters={'I_ab_npp_Young':I_ab_npp_Young,'I_root_npp_Young_top':I_root_npp_Young_top,'I_root_npp_Young_sub':I_root_npp_Young_sub,
				'I_Young_top_down':I_Young_top_down,'I_Young_sub_down':I_Young_sub_down,'I_Old_top_down':I_Old_top_down,'I_Old_sub_down':I_Old_sub_down,
				'k_ab_npp':k_ab_npp,'k_Young_top':k_Young_top,'k_Young_sub':k_Young_sub,'k_Old_top':k_Old_top,'k_Old_sub':k_Old_sub,
				'e_ab_npp_Old':e_ab_npp_Old,'e_Young_Old_top':e_Young_Old_top,'e_Young_Old_sub':e_Young_Old_sub,'pmax':pmax,'Km':Km}
	return parameters

def CDynamics_simulation(NPP,fBNPP,fr20,fr20_100,C_Young_top,C_Old_top,C_Young_sub,C_Old_sub,parameters,simulation_year,is_sensitive_test):
	Cab_npp,Croot20_npp,Croot20_100_npp=NPP_Ag_Bg_allocation(NPP,fBNPP,fr20,fr20_100)
	C_Young_top_dynamics=[C_Young_top]
	C_Old_top_dynamics=[C_Old_top]	
	C_Young_Old_top_all=[C_Young_top+C_Old_top]
	C_Young_sub_dynamics=[C_Young_sub]
	C_Old_sub_dynamics=[C_Old_sub]	
	C_Young_Old_sub_all=[C_Young_sub+C_Old_sub]
	I_ab_npp_Young=parameters['I_ab_npp_Young']	
	I_root_npp_Young_top=parameters['I_root_npp_Young_top']
	I_root_npp_Young_sub=parameters['I_root_npp_Young_sub']	
	I_Young_top_down=parameters['I_Young_top_down']
	I_Young_sub_down=parameters['I_Young_sub_down']
	I_Old_top_down=parameters['I_Old_top_down']
	I_Old_sub_down=parameters['I_Old_sub_down']	
	k_ab_npp=parameters['k_ab_npp']
	k_Young_top=parameters['k_Young_top']
	k_Young_sub=parameters['k_Young_sub']
	k_Old_top=parameters['k_Old_top']
	k_Old_sub=parameters['k_Old_sub']	
	e_ab_npp_Old=parameters['e_ab_npp_Old']
	e_Young_Old_top=parameters['e_Young_Old_top']
	e_Young_Old_sub=parameters['e_Young_Old_sub']
	pmax=parameters['pmax']
	Km=parameters['Km']	
	C_Young_top_dynamics=[C_Young_top]
	C_Old_top_dynamics=[C_Old_top]	
	C_Young_Old_top_all=[C_Young_top+C_Old_top]	
	C_Young_sub_dynamics=[C_Young_sub]
	C_Old_sub_dynamics=[C_Old_sub]	
	C_Young_Old_sub_all=[C_Young_sub+C_Old_sub]	
	for i in range(simulation_year):
		C_Young_top_rate=I_ab_npp_Young*Cab_npp+Croot20_npp*I_root_npp_Young_top-I_Young_top_down*C_Young_top-(1-I_Young_top_down)*C_Young_top*k_Young_top
		C_Old_top_rate=Cab_npp*(1-I_ab_npp_Young)*e_ab_npp_Old*k_ab_npp+(1-I_Young_top_down)*C_Young_top*e_Young_Old_top*k_Young_top-I_Old_top_down*C_Old_top-(1-I_Old_top_down)*C_Old_top*k_Old_top*(1+calculate_PE(pmax,C_Young_top,Km))   
		C_Young_sub_rate=I_Young_top_down*C_Young_top+Croot20_100_npp*I_root_npp_Young_sub-I_Young_sub_down*C_Young_sub-(1-I_Young_sub_down)*C_Young_sub*k_Young_sub
		C_Old_sub_rate=I_Old_top_down*C_Old_top+(1-I_Young_sub_down)*C_Young_sub*e_Young_Old_sub*k_Young_sub-I_Old_sub_down*C_Old_sub-(1-I_Old_sub_down)*C_Old_sub*k_Old_sub*(1+calculate_PE(pmax,C_Young_sub,Km)) 
		C_Young_top+=C_Young_top_rate
		C_Old_top+=C_Old_top_rate
		C_Young_sub+=C_Young_sub_rate
		C_Old_sub+=C_Old_sub_rate		 
		C_Young_top_dynamics.append(C_Young_top)
		C_Old_top_dynamics.append(C_Old_top)
		C_Young_Old_top_all.append(C_Young_top+C_Old_top)
		C_Young_sub_dynamics.append(C_Young_sub)
		C_Old_sub_dynamics.append(C_Old_sub)
		C_Young_Old_sub_all.append(C_Young_sub+C_Old_sub)		
	if is_sensitive_test:
		return C_Young_Old_top_all,C_Young_Old_sub_all
	else:
		return C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all

def SoilC_Dynamics_4pools_to_optimize(spin_up_year,NPP,fBNPP,fr20,fr20_100,I_Old_top_down, I_Old_sub_down, k_ab_npp, k_Old_top, k_Old_sub, e_ab_npp_Old):
	Cab_npp,Croot20_npp,Croot20_100_npp=NPP_Ag_Bg_allocation(NPP,fBNPP,fr20,fr20_100)
	C_Young_top,C_Old_top=0,0
	C_Young_sub,C_Old_sub=0,0
	I_ab_npp_Young=0.04
	I_root_npp_Young_top=0.03
	I_root_npp_Young_sub=0.02
	I_Young_top_down=0.04
	I_Young_sub_down=0.01	
	k_Young_top_down=0.02
	k_Young_sub_down=0.001
	e_Young_Old_top=0.2
	e_Young_Old_sub=0.1	
	pmax=-0.5
	Km=500
	C_Young_top_dynamics=[C_Young_top]
	C_Old_top_dynamics=[C_Old_top]	
	C_Young_Old_top_all=[C_Young_top+C_Old_top]
	C_Young_sub_dynamics=[C_Young_sub]
	C_Old_sub_dynamics=[C_Old_sub]	
	C_Young_Old_sub_all=[C_Young_sub+C_Old_sub]	
	for i in range(spin_up_year):
		C_Young_top_rate=I_ab_npp_Young*Cab_npp+Croot20_npp*I_root_npp_Young_top-I_Young_top_down*C_Young_top-(1-I_Young_top_down)*C_Young_top*k_Young_top_down
		C_Old_top_rate=Cab_npp*(1-I_ab_npp_Young)*e_ab_npp_Old*k_ab_npp+(1-I_Young_top_down)*C_Young_top*e_Young_Old_top*k_Young_top_down-I_Old_top_down*C_Old_top-(1-I_Old_top_down)*C_Old_top*k_Old_top*(1+calculate_PE(pmax,C_Young_top,Km))   
		C_Young_sub_rate=I_Young_top_down*C_Young_top+Croot20_100_npp*I_root_npp_Young_sub-I_Young_sub_down*C_Young_sub-(1-I_Young_sub_down)*C_Young_sub*k_Young_sub_down
		C_Old_sub_rate=I_Old_top_down*C_Old_top+(1-I_Young_sub_down)*C_Young_sub*e_Young_Old_sub*k_Young_sub_down-I_Old_sub_down*C_Old_sub-(1-I_Old_sub_down)*C_Old_sub*k_Old_sub*(1+calculate_PE(pmax,C_Young_sub,Km)) 
		C_Young_top+=C_Young_top_rate
		C_Old_top+=C_Old_top_rate
		C_Young_sub+=C_Young_sub_rate
		C_Old_sub+=C_Old_sub_rate	
		C_Young_top_dynamics.append(C_Young_top)
		C_Old_top_dynamics.append(C_Old_top)
		C_Young_Old_top_all.append(C_Young_top+C_Old_top)
		C_Young_sub_dynamics.append(C_Young_sub)
		C_Old_sub_dynamics.append(C_Old_sub)
		C_Young_Old_sub_all.append(C_Young_sub+C_Old_sub)	
	return C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all


def objective_function(params, spin_up_year, NPP, fBNPP, fr20, fr20_100, observed_C_Young_Old_top_all, observed_C_Young_Old_sub_all):
	I_Old_top_down, I_Old_sub_down, k_ab_npp, k_Old_top, k_Old_sub, e_ab_npp_Old = params
	C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all = SoilC_Dynamics_4pools_to_optimize(
		spin_up_year, NPP, fBNPP, fr20, fr20_100, I_Old_top_down, I_Old_sub_down, k_ab_npp, k_Old_top, k_Old_sub, e_ab_npp_Old
	)
	mse_top = np.mean((np.array(C_Young_Old_top_all[-1]) - np.array(observed_C_Young_Old_top_all))**2)
	mse_sub = np.mean((np.array(C_Young_Old_sub_all[-1]) - np.array(observed_C_Young_Old_sub_all))**2)	
	return np.abs(mse_top) + np.abs(mse_sub)

def parameter_optimize_PSO(spin_up_year,NPP, fBNPP, fr20, fr20_100, observed_C_Young_Old_top_all,observed_C_Young_Old_sub_all):

	lb = [0.01, 0.01, 0.1, 0.000001, 0.000001, 0.2]  
	ub = [0.05, 0.05, 1.0, 0.1, 0.1, 0.8]	
	optimized_params, fopt = pso(
		objective_function, lb, ub, args=(spin_up_year, NPP, fBNPP, fr20, fr20_100, observed_C_Young_Old_top_all, observed_C_Young_Old_sub_all),
		swarmsize=50, maxiter=200, minstep=1e-3*3.0, minfunc=1e-3*3.0
	)
	return optimized_params

def para_optimize_for_spatial_array_PSO(NPP_raster_file,SOC_20_raster_file,SOC_20_100_raster_file,fBNPP_f20_f20_100_raster_file_list,spin_up_year,opti_para_all_data_txt_filepath,start_line,end_line):
	[NPP_im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]=read_rasterfile_setnan(NPP_raster_file)
	SOC_20_im_data=read_rasterfile_setnan(SOC_20_raster_file)[0]*1000.0
	SOC_20_100_im_data=read_rasterfile_setnan(SOC_20_100_raster_file)[0]*1000.0
	fBNPP_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[0])[0]
	f20_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[1])[0]
	f20_100_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[2])[0]
	x,y=np.where((NPP_im_data[start_line:end_line,:]>0) & (SOC_20_im_data[start_line:end_line,:]>0) & (SOC_20_100_im_data[start_line:end_line,:]>0) & (fBNPP_im_data[start_line:end_line,:]>0) & (f20_im_data[start_line:end_line,:]>0) & (f20_100_im_data[start_line:end_line,:]>0))
	if x.shape[0]>0:
		current_line=x[0]+start_line   
		current_line_have_data_list=[]
		print('line '+str(current_line)+' start')
		this_line_num=0
		this_line_points=x[x==current_line-start_line].shape[0]
		print('line '+str(current_line)+': '+str(this_line_points)+' points')
		for k in range(x.shape[0]):
			i,j=x[k]+start_line,y[k]
			this_line_num+=1
			if i==current_line:
				this_cell_optimized_params=parameter_optimize_PSO(spin_up_year,NPP_im_data[i,j], fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j], SOC_20_im_data[i,j],SOC_20_100_im_data[i,j])
				C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all=SoilC_Dynamics_4pools_to_optimize(spin_up_year,NPP_im_data[i,j], fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j],this_cell_optimized_params[0],this_cell_optimized_params[1],this_cell_optimized_params[2],this_cell_optimized_params[3],this_cell_optimized_params[4],this_cell_optimized_params[5])
				PBias_list=[pbias(C_Young_Old_top_all[-1],SOC_20_im_data[i,j]),pbias(C_Young_Old_sub_all[-1],SOC_20_100_im_data[i,j])]
				current_year_i_j_all_data=list(np.hstack([[str(int(i)),str(int(j))],np.around(this_cell_optimized_params,6),
															np.around([C_Young_Old_top_all[-1],C_Young_Old_sub_all[-1]],4),
															np.around(PBias_list,4)]))
				current_line_have_data_list.append(current_year_i_j_all_data)				
				print('This line Progress: '+f"{round(this_line_num*100.0/this_line_points,2):.2f}"+'% | Total Progress: '+f"{round((k+1)*100.0/x.shape[0],2):.2f}"+'%',end='\r')
				if k==x.shape[0]-1:
					current_line_opti_para_txt_file=opti_para_all_data_txt_filepath+'grid_line['+str(current_line)+']_opti_para_all_data.txt'
					write_2dim_list_to_txt_file(current_line_opti_para_txt_file,current_line_have_data_list)
					print('Last line: '+str(round((k+1)*100.0/x.shape[0],2))+"%")
			elif len(current_line_have_data_list)!=0 and i!=current_line:				
				current_line_opti_para_txt_file=opti_para_all_data_txt_filepath+'grid_line['+str(current_line)+']_opti_para_all_data.txt'
				write_2dim_list_to_txt_file(current_line_opti_para_txt_file,current_line_have_data_list)
				current_line=i
				print('line '+str(current_line)+' start')
				this_line_num=1
				this_line_points=x[x==current_line-start_line].shape[0]
				print('line '+str(current_line)+':'+str(x[x==current_line-start_line].shape[0])+' points')	
				current_line_have_data_list=[]
				this_cell_optimized_params=parameter_optimize_PSO(spin_up_year,NPP_im_data[i,j], fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j], SOC_20_im_data[i,j],SOC_20_100_im_data[i,j])
				C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all=SoilC_Dynamics_4pools_to_optimize(spin_up_year,NPP_im_data[i,j], fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j],this_cell_optimized_params[0],this_cell_optimized_params[1],this_cell_optimized_params[2],this_cell_optimized_params[3],this_cell_optimized_params[4],this_cell_optimized_params[5])
				PBias_list=[pbias(C_Young_Old_top_all[-1],SOC_20_im_data[i,j]),pbias(C_Young_Old_sub_all[-1],SOC_20_100_im_data[i,j])]
				current_year_i_j_all_data=list(np.hstack([[str(int(i)),str(int(j))],np.around(this_cell_optimized_params,6),
															np.around([C_Young_Old_top_all[-1],C_Young_Old_sub_all[-1]],4),
															np.around(PBias_list,4)]))
				current_line_have_data_list.append(current_year_i_j_all_data)
				print('This line Progress: '+f"{round(this_line_num*100.0/this_line_points,2):.2f}"+'% | Total Progress: '+f"{round((k+1)*100.0/x.shape[0],2):.2f}"+'%',end='\r')
		print('opti para, pbias, and sim result rasters write done')


def read_txt_convert_raster(opti_para_all_data_txt_filepath,reference_raster_filepath,output_raster_path):
	all_data_txt_list=get_all_type_files(opti_para_all_data_txt_filepath,'.txt')
	[im_data,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]=read_rasterfile_setnan(reference_raster_filepath)
	estimated_soc20=np.empty([im_height,im_width])
	estimated_soc100=np.empty([im_height,im_width])
	estimated_soc20[:]=-999.0
	estimated_soc100[:]=-999.0
	bias_soc20=np.empty([im_height,im_width])
	bias_soc100=np.empty([im_height,im_width])
	bias_soc20[:]=-999.0
	bias_soc100[:]=-999.0
	I_Old_top_down=np.full([im_height,im_width],-999.0)
	I_Old_sub_down=np.full([im_height,im_width],-999.0)
	k_ab_npp=np.full([im_height,im_width],-999.0)
	k_Old_top=np.full([im_height,im_width],-999.0)
	k_Old_sub=np.full([im_height,im_width],-999.0)
	e_ab_npp_Old=np.full([im_height,im_width],-999.0)
	for each in all_data_txt_list:
		all_data_array=read_2dim_txt(each)
		for each_line in all_data_array:
			row=int(each_line[0])
			column=int(each_line[1])
			estimated_soc20[row,column]=each_line[-4]
			estimated_soc100[row,column]=each_line[-3]
			bias_soc20[row,column]=each_line[-2]
			bias_soc100[row,column]=each_line[-1]
			#参数文件
			I_Old_top_down[row,column]=each_line[2]
			I_Old_sub_down[row,column]=each_line[3]
			k_ab_npp[row,column]=each_line[4]
			k_Old_top[row,column]=each_line[5]
			k_Old_sub[row,column]=each_line[6]
			e_ab_npp_Old[row,column]=each_line[7]
		print('converting...'+str(round((all_data_txt_list.index(each)+1)*100.0/len(all_data_txt_list),2))+"%",end='\r')
	estimated_soc20_file=output_raster_path+'estimated_SOC20.tif'
	estimated_soc100_file=output_raster_path+'estimated_SOC20_100.tif'
	bias_soc20_file=output_raster_path+'bias_SOC20.tif'
	bias_soc100_file=output_raster_path+'bias_SOC20_100.tif'
	I_Old_top_down_file=output_raster_path+'I_Old_top_down.tif'
	I_Old_sub_down_file=output_raster_path+'I_Old_sub_down.tif'
	k_ab_npp_file=output_raster_path+'k_ab_npp.tif'
	k_Old_top_file=output_raster_path+'k_Old_top.tif'
	k_Old_sub_file=output_raster_path+'k_Old_sub.tif'
	e_ab_npp_Old_file=output_raster_path+'e_ab_npp_Old.tif'
	write_rasterfile(estimated_soc20_file,estimated_soc20,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(estimated_soc100_file,estimated_soc100,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(bias_soc20_file,bias_soc20,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(bias_soc100_file,bias_soc100,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(I_Old_top_down_file,I_Old_top_down,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(I_Old_sub_down_file,I_Old_sub_down,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(k_ab_npp_file,k_ab_npp,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(k_Old_top_file,k_Old_top,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(k_Old_sub_file,k_Old_sub,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	write_rasterfile(e_ab_npp_Old_file,e_ab_npp_Old,im_width,im_height,im_bands,im_geotrans,im_proj,-999.0)
	print('raster write done')
	

def calculate_K_ratio(Temp_first_year, Temp_this_year):
	if Temp_first_year<=35:
		f0_CANDY=2.1**((Temp_first_year-35.0)/10.0)
	else:
		f0_CANDY=1.0
	if Temp_this_year<=35:
		f1_CANDY=2.1**((Temp_this_year-35.0)/10.0)
	else:
		f1_CANDY=1
	K_ratio_CANDY=f1_CANDY/f0_CANDY
	K_ratio_CASA_CNP=(1.71**((Temp_this_year-35.0)/10.0))/(1.71**((Temp_first_year-35.0)/10.0))
	K_ratio_CENTURY=(0.56+0.465*np.arctan(0.097*(Temp_this_year-15.7)))/(0.56+0.465*np.arctan(0.097*(Temp_first_year-15.7)))
	T1_first_year=(45-Temp_first_year)/10.0
	T2_first_year=np.exp(0.076*(1-T1_first_year**2.63))
	f0_GFDL_ESMM2M=(T1_first_year**0.2)*T2_first_year
	T1_this_year=(45-Temp_this_year)/10.0
	T2_this_year=np.exp(0.076*(1-T1_this_year**2.63))
	f1_GFDL_ESMM2M=(T1_this_year**0.2)*T2_this_year
	K_ratio_GFDL_ESMM2M=f1_GFDL_ESMM2M/f0_GFDL_ESMM2M
	K_ratio_K2000=(np.exp(3.36*(Temp_this_year-40)/(Temp_this_year+46.05))/4.71)/(np.exp(3.36*(Temp_first_year-40)/(Temp_first_year+46.05))/4.71)
	K_ratio_LPJ=(np.exp(308.56*(1/56.02-1/(Temp_this_year+46.02)))/4.71)/(np.exp(308.56*(1/56.02-1/(Temp_first_year+46.02)))/4.71)
	K_ratio_PnET_CN=(0.68*np.exp(0.1*(Temp_this_year-7.1))/12.64)/(0.68*np.exp(0.1*(Temp_first_year-7.1))/12.64)
	K_ratio_ROTHC=(47.9/(1+np.exp(106/(Temp_this_year+18.3))))/(47.9/(1+np.exp(106/(Temp_first_year+18.3))))
	return [K_ratio_CANDY,K_ratio_CASA_CNP,K_ratio_CENTURY,K_ratio_GFDL_ESMM2M,K_ratio_K2000,K_ratio_LPJ,K_ratio_PnET_CN,K_ratio_ROTHC]


def calculate_fT_K_ratio(Temp_first_year, Temp_this_year):
	f0_Century1=-9.0*(10**(-7))*(Temp_first_year**4) + 2*(10**(-5))*(Temp_first_year**3) + 0.0009*(Temp_first_year**2) + 0.0067*Temp_first_year + 0.0105
	f1_Century1=-9.0*(10**(-7))*(Temp_this_year**4) + 2*(10**(-5))*(Temp_this_year**3) + 0.0009*(Temp_this_year**2) + 0.0067*Temp_this_year + 0.0105
	K_ratio_Century1=f1_Century1/f0_Century1
	f0_Century2=2*(10**(-9))*(Temp_first_year**6) - (10**(-7))*(Temp_first_year**5) + 2*(10**(-7))*(Temp_first_year**4) + 0.0001*(Temp_first_year**3) + 0.0013*(Temp_first_year**2) + 0.0026*Temp_first_year + 0.0208
	f1_Century2=2*(10**(-9))*(Temp_this_year**6) - (10**(-7))*(Temp_this_year**5) + 2*(10**(-7))*(Temp_this_year**4) + 0.0001*(Temp_this_year**3) + 0.0013*(Temp_this_year**2) + 0.0026*Temp_this_year + 0.0208
	K_ratio_Century2=f1_Century2/f0_Century2
	f0_Daycent1=2*(10**(-8))*(Temp_first_year**5) - (10**(-7))*(Temp_first_year**4) + 3*(10**(-6))*(Temp_first_year**3) + 0.0005*(Temp_first_year**2) + 0.0078*Temp_first_year + 0.0665
	f1_Daycent1=2*(10**(-8))*(Temp_this_year**5) - (10**(-7))*(Temp_this_year**4) + 3*(10**(-6))*(Temp_this_year**3) + 0.0005*(Temp_this_year**2) + 0.0078*Temp_this_year + 0.0665
	K_ratio_Daycent1=f1_Daycent1/f0_Daycent1
	f0_Daycent2=4*(10**(-10))*(Temp_first_year**6) - (10**(-8))*(Temp_first_year**5) - 9*(10**(-7))*(Temp_first_year**4) + 2*(10**(-5))*(Temp_first_year**3) + 0.0009*(Temp_first_year**2) + 0.0152*Temp_first_year + 0.0922
	f1_Daycent2=4*(10**(-10))*(Temp_this_year**6) - (10**(-8))*(Temp_this_year**5) - 9*(10**(-7))*(Temp_this_year**4) + 2*(10**(-5))*(Temp_this_year**3) + 0.0009*(Temp_this_year**2) + 0.0152*Temp_this_year + 0.0922
	K_ratio_Daycent2=f1_Daycent2/f0_Daycent2
	f0_Q10_2=2*(10**(-8))*(Temp_first_year**5) + 2*(10**(-7))*(Temp_first_year**4) + 2*(10**(-5))*(Temp_first_year**3) + 0.0013*(Temp_first_year**2) + 0.0352*Temp_first_year + 0.4949
	f1_Q10_2=2*(10**(-8))*(Temp_this_year**5) + 2*(10**(-7))*(Temp_this_year**4) + 2*(10**(-5))*(Temp_this_year**3) + 0.0013*(Temp_this_year**2) + 0.0352*Temp_this_year + 0.4949
	K_ratio_Q10_2=f1_Q10_2/f0_Q10_2
	f0_Q10_14=4*(10**(-10))*(Temp_first_year**5) + 4*(10**(-8))*(Temp_first_year**4) + 4*(10**(-6))*(Temp_first_year**3) + 0.0004*(Temp_first_year**2) + 0.024*Temp_first_year + 0.7142
	f1_Q10_14=4*(10**(-10))*(Temp_this_year**5) + 4*(10**(-8))*(Temp_this_year**4) + 4*(10**(-6))*(Temp_this_year**3) + 0.0004*(Temp_this_year**2) + 0.024*Temp_this_year + 0.7142
	K_ratio_Q10_14=f1_Q10_14/f0_Q10_14
	f0_Lloyd_Taylor=1*(10**(-10))*(Temp_first_year**5) - 4*(10**(-7))*(Temp_first_year**4) + 3*(10**(-5))*(Temp_first_year**3) + 0.0023*(Temp_first_year**2) + 0.0438*Temp_first_year + 0.2938
	f1_Lloyd_Taylor=1*(10**(-10))*(Temp_this_year**5) - 4*(10**(-7))*(Temp_this_year**4) + 3*(10**(-5))*(Temp_this_year**3) + 0.0023*(Temp_this_year**2) + 0.0438*Temp_this_year + 0.2938
	K_ratio_Lloyd_Taylor=f1_Lloyd_Taylor/f0_Lloyd_Taylor
	f0_Kirschbaum=5*(10**(-10))*(Temp_first_year**6) - 4*(10**(-8))*(Temp_first_year**5) + 6*(10**(-9))*(Temp_first_year**4) + 4*(10**(-5))*(Temp_first_year**3) + 0.0005*(Temp_first_year**2) + 0.003*Temp_first_year + 0.0222
	f1_Kirschbaum=5*(10**(-10))*(Temp_this_year**6) - 4*(10**(-8))*(Temp_this_year**5) + 6*(10**(-9))*(Temp_this_year**4) + 4*(10**(-5))*(Temp_this_year**3) + 0.0005*(Temp_this_year**2) + 0.003*Temp_this_year + 0.0222
	K_ratio_Kirschbaum=f1_Kirschbaum/f0_Kirschbaum
	f0_Demeter=1*(10**(-8))*(Temp_first_year**5) + 1*(10**(-7))*(Temp_first_year**4) + 1*(10**(-5))*(Temp_first_year**3) + 0.0007*(Temp_first_year**2) + 0.0176*Temp_first_year + 0.2475
	f1_Demeter=1*(10**(-8))*(Temp_this_year**5) + 1*(10**(-7))*(Temp_this_year**4) + 1*(10**(-5))*(Temp_this_year**3) + 0.0007*(Temp_this_year**2) + 0.0176*Temp_this_year + 0.2475
	K_ratio_Demeter=f1_Demeter/f0_Demeter
	f0_Standcarb=3*(10**(-9))*(Temp_first_year**5) + 5*(10**(-7))*(Temp_first_year**4) + 3*(10**(-5))*(Temp_first_year**3) + 0.0012*(Temp_first_year**2) + 0.0346*Temp_first_year + 0.5011
	f1_Standcarb=3*(10**(-9))*(Temp_this_year**5) + 5*(10**(-7))*(Temp_this_year**4) + 3*(10**(-5))*(Temp_this_year**3) + 0.0012*(Temp_this_year**2) + 0.0346*Temp_this_year + 0.5011
	K_ratio_Standcarb=f1_Standcarb/f0_Standcarb
	return [K_ratio_Century1,K_ratio_Century2,K_ratio_Daycent1,K_ratio_Daycent2,K_ratio_Q10_2,K_ratio_Q10_14,K_ratio_Lloyd_Taylor,K_ratio_Kirschbaum,K_ratio_Demeter,K_ratio_Standcarb]

def calculate_fW_K_ratio(SM_first_year, SM_this_year):
	f0_Century=4.7379*(SM_first_year**4) -12.372*(SM_first_year**3) + 9.6178*(SM_first_year**2) - 1.0724*SM_first_year + 0.0742
	f1_Century=4.7379*(SM_this_year**4) -12.372*(SM_this_year**3) + 9.6178*(SM_this_year**2) - 1.0724*SM_this_year + 0.0742
	K_ratio_Century=f1_Century/f0_Century
	f0_Daycent=11.595*(SM_first_year**4) - 27.636*(SM_first_year**3) + 18.022*(SM_first_year**2) - 1.7956*SM_first_year + 0.0428
	f1_Daycent=11.595*(SM_this_year**4) - 27.636*(SM_this_year**3) + 18.022*(SM_this_year**2) - 1.7956*SM_this_year + 0.0428
	K_ratio_Daycent=f1_Daycent/f0_Daycent
	f0_Demeter=0.75*SM_first_year + 0.25
	f1_Demeter=0.75*SM_this_year + 0.25
	K_ratio_Demeter=f1_Demeter/f0_Demeter
	f0_Standcarb=-18.878*(SM_first_year**4) + 38.896*(SM_first_year**3) - 29.765*(SM_first_year**2) + 9.9718*SM_first_year - 0.2258
	f1_Standcarb=-18.878*(SM_this_year**4) + 38.896*(SM_this_year**3) - 29.765*(SM_this_year**2) + 9.9718*SM_this_year - 0.2258
	K_ratio_Standcarb=f1_Standcarb/f0_Standcarb
	f0_Candy=-1.6388*(SM_first_year**6) + 1.8458*(SM_first_year**5) + 1.8011*(SM_first_year**4) -0.4371*(SM_first_year**3) - 4.7698*(SM_first_year**2) + 4.199*SM_first_year - 0.0024
	f1_Candy=-1.6388*(SM_this_year**6) + 1.8458*(SM_this_year**5) + 1.8011*(SM_this_year**4) -0.4371*(SM_this_year**3) - 4.7698*(SM_this_year**2) + 4.199*SM_this_year - 0.0024
	K_ratio_Candy=f1_Candy/f0_Candy
	if SM_first_year<=0.16:
		f0_Gompertz=154.2*(SM_first_year**3) -81.248*(SM_first_year**2) + 15.112*SM_first_year + 0.0337
	elif SM_first_year>0.16:
		f0_Gompertz=1.0
	if SM_this_year<=0.16:
		f1_Gompertz=154.2*(SM_this_year**3) -81.248*(SM_this_year**2) + 15.112*SM_this_year + 0.0337
	elif SM_this_year>0.16:
		f1_Gompertz=1.0
	K_ratio_Gompertz=f1_Gompertz/f0_Gompertz
	f0_Myers=-1.0141*(SM_first_year**2) + 2.0093*SM_first_year - 0.0014
	f1_Myers=-1.0141*(SM_this_year**2) + 2.0093*SM_this_year - 0.0014
	K_ratio_Myers=f1_Myers/f0_Myers
	f0_Moyano=-2.4095*(SM_first_year**2) + 3.0973*SM_first_year + 0.0026
	f1_Moyano=-2.4095*(SM_this_year**2) + 3.0973*SM_this_year + 0.0026
	K_ratio_Moyano=f1_Moyano/f0_Moyano
	if SM_first_year<=0.6:
		f0_Skopp=1.7824*SM_first_year - 0.0818
	elif SM_first_year>0.6:
		f0_Skopp=-2.3066*SM_first_year +2.3561
	if SM_this_year<=0.6:
		f1_Skopp=1.7824*SM_this_year - 0.0818
	elif SM_this_year>0.6:
		f1_Skopp=-2.3066*SM_this_year +2.3561
	K_ratio_Skopp=f1_Skopp/f0_Skopp
	return [K_ratio_Century,K_ratio_Daycent,K_ratio_Demeter,K_ratio_Standcarb,K_ratio_Candy,K_ratio_Gompertz,K_ratio_Myers,K_ratio_Moyano,K_ratio_Skopp]

def SoilC_Dynamics_4pools_for_simulation(NPP_series,Temp_series, SM_top_series, SM_sub_series, fBNPP,fr20,fr20_100,parameters_list, initial_carbon_pool_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario):
	[I_Old_top_down, I_Old_sub_down, k_ab_npp, k_Old_top, k_Old_sub, e_ab_npp_Old]=parameters_list
	[C_Young_top,C_Old_top,C_Young_sub,C_Old_sub]=initial_carbon_pool_list
	I_ab_npp_Young=0.04
	I_root_npp_Young_top=0.03
	I_root_npp_Young_sub=0.02
	I_Young_top_down=0.04
	I_Young_sub_down=0.01	
	k_Young_top_down=0.02
	k_Young_sub_down=0.001
	e_Young_Old_top=0.2
	e_Young_Old_sub=0.1	
	pmax=-0.5
	Km=500	
	C_Young_top_dynamics=[C_Young_top]
	C_Old_top_dynamics=[C_Old_top]	
	C_Young_Old_top_all=[C_Young_top+C_Old_top]
	C_Young_sub_dynamics=[C_Young_sub]
	C_Old_sub_dynamics=[C_Old_sub]	
	C_Young_Old_sub_all=[C_Young_sub+C_Old_sub]	
	simulation_year=NPP_series.shape[0]
	for i in range(simulation_year)[1:]:
		Cab_npp,Croot20_npp,Croot20_100_npp=NPP_Ag_Bg_allocation(NPP_series[i],fBNPP,fr20,fr20_100)
		Temp_first_year, Temp_this_year=Temp_series[0],Temp_series[i]
		SM_top_first_year, SM_top_this_year=SM_top_series[0],SM_top_series[i]
		SM_sub_first_year, SM_sub_this_year=SM_sub_series[0],SM_sub_series[i][K_ratio_Century1,K_ratio_Century2,K_ratio_Daycent1,K_ratio_Daycent2,K_ratio_Q10_2,K_ratio_Q10_14,K_ratio_Lloyd_Taylor,K_ratio_Kirschbaum,K_ratio_Demeter,K_ratio_Standcarb]=calculate_fT_K_ratio(Temp_first_year, Temp_this_year)
		fT_K_ratio_list=[K_ratio_Century1,K_ratio_Century2,K_ratio_Daycent1,K_ratio_Daycent2,K_ratio_Q10_2,K_ratio_Q10_14,K_ratio_Lloyd_Taylor,K_ratio_Kirschbaum,K_ratio_Demeter,K_ratio_Standcarb]
		[K_ratio_Century,K_ratio_Daycent,K_ratio_Demeter,K_ratio_Standcarb,K_ratio_Candy,K_ratio_Gompertz,K_ratio_Myers,K_ratio_Moyano,K_ratio_Skopp]=calculate_fW_K_ratio(SM_top_first_year, SM_top_this_year)
		fW_top_K_ratio_list=[K_ratio_Century,K_ratio_Daycent,K_ratio_Demeter,K_ratio_Standcarb,K_ratio_Candy,K_ratio_Gompertz,K_ratio_Myers,K_ratio_Moyano,K_ratio_Skopp]
		[K_ratio_Century,K_ratio_Daycent,K_ratio_Demeter,K_ratio_Standcarb,K_ratio_Candy,K_ratio_Gompertz,K_ratio_Myers,K_ratio_Moyano,K_ratio_Skopp]=calculate_fW_K_ratio(SM_sub_first_year, SM_sub_this_year)
		fW_sub_K_ratio_list=[K_ratio_Century,K_ratio_Daycent,K_ratio_Demeter,K_ratio_Standcarb,K_ratio_Candy,K_ratio_Gompertz,K_ratio_Myers,K_ratio_Moyano,K_ratio_Skopp]
		k_Young_top_down=get_new_fT_fW_K_factor(k_Young_top_down,K_driver_scenario,fT_K_ratio_list,fW_top_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index)
		if k_Young_top_down>1:
			k_Young_top_down=1
		elif k_Young_top_down<0.01:
			k_Young_top_down=0.01
		k_ab_npp=get_new_fT_fW_K_factor(k_ab_npp,K_driver_scenario,fT_K_ratio_list,fW_top_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index)
		if k_ab_npp>1:
			k_ab_npp=1
		elif k_ab_npp<0.1:
			k_ab_npp=0.1
		k_Old_top=get_new_fT_fW_K_factor(k_Old_top,K_driver_scenario,fT_K_ratio_list,fW_top_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index)
		if k_Old_top>0.1:
			k_Old_top=0.1
		elif k_Old_top<0.000001:
			k_Old_top=0.000001
		k_Young_sub_down=get_new_fT_fW_K_factor(k_Young_sub_down,K_driver_scenario,fT_K_ratio_list,fW_sub_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index)
		if k_Young_sub_down>1:
			k_Young_sub_down=1
		elif k_Young_sub_down<0.01:
			k_Young_sub_down=0.01
		k_Old_sub=get_new_fT_fW_K_factor(k_Old_sub,K_driver_scenario,fT_K_ratio_list,fW_sub_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index)
		if k_Old_sub>0.1:
			k_Old_sub=0.1
		elif k_Old_sub<0.000001:
			k_Old_sub=0.000001
		C_Young_top_rate=I_ab_npp_Young*Cab_npp+Croot20_npp*I_root_npp_Young_top-I_Young_top_down*C_Young_top-(1-I_Young_top_down)*C_Young_top*k_Young_top_down
		C_Old_top_rate=Cab_npp*(1-I_ab_npp_Young)*e_ab_npp_Old*k_ab_npp+(1-I_Young_top_down)*C_Young_top*e_Young_Old_top*k_Young_top_down-I_Old_top_down*C_Old_top-(1-I_Old_top_down)*C_Old_top*k_Old_top*(1+calculate_PE(pmax,C_Young_top,Km))   
		C_Young_sub_rate=I_Young_top_down*C_Young_top+Croot20_100_npp*I_root_npp_Young_sub-I_Young_sub_down*C_Young_sub-(1-I_Young_sub_down)*C_Young_sub*k_Young_sub_down
		C_Old_sub_rate=I_Old_top_down*C_Old_top+(1-I_Young_sub_down)*C_Young_sub*e_Young_Old_sub*k_Young_sub_down-I_Old_sub_down*C_Old_sub-(1-I_Old_sub_down)*C_Old_sub*k_Old_sub*(1+calculate_PE(pmax,C_Young_sub,Km))
		C_Young_top+=C_Young_top_rate
		C_Old_top+=C_Old_top_rate
		C_Young_sub+=C_Young_sub_rate
		C_Old_sub+=C_Old_sub_rate
		C_Young_top_dynamics.append(C_Young_top)
		C_Old_top_dynamics.append(C_Old_top)
		C_Young_Old_top_all.append(C_Young_top+C_Old_top)
		C_Young_sub_dynamics.append(C_Young_sub)
		C_Old_sub_dynamics.append(C_Old_sub)
		C_Young_Old_sub_all.append(C_Young_sub+C_Old_sub)	
	return C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all


def get_new_fT_fW_K_factor(K_factor,K_driver_scenario,fT_K_ratio_list,fW_K_ratio_list,fT_K_ratio_method_index,fW_K_ratio_method_index):
	if K_driver_scenario==1:
		return K_factor*fT_K_ratio_list[fT_K_ratio_method_index]
	elif K_driver_scenario==2:
		return K_factor*fW_K_ratio_list[fW_K_ratio_method_index]
	elif K_driver_scenario==3:
		return K_factor*fT_K_ratio_list[fT_K_ratio_method_index]*fW_K_ratio_list[fW_K_ratio_method_index]

def get_new_fT_K_factor(K_factor,K_driver_scenario,fT_K_ratio_list,fT_K_ratio_method_index):
	if K_driver_scenario==1:
		return K_factor*fT_K_ratio_list[fT_K_ratio_method_index]

def simulate_SOC_based_on_NPP_Temp_SM(NPP_time_series_filepath,parameter_filepath,Temp_series_filepath,SM_top_series_filepath, SM_sub_series_filepath, fBNPP_f20_f20_100_raster_file_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario,Output_SOC_series_filepath):
	I_Old_top_down_file=parameter_filepath+'I_Old_top_down.tif'
	I_Old_sub_down_file=parameter_filepath+'I_Old_sub_down.tif'
	k_ab_npp_file=parameter_filepath+'k_ab_npp.tif'
	k_Old_top_file=parameter_filepath+'k_Old_top.tif'
	k_Old_sub_file=parameter_filepath+'k_Old_sub.tif'
	e_ab_npp_Old_file=parameter_filepath+'e_ab_npp_Old.tif'
	[I_Old_top_down,im_width,im_height,im_bands,im_geotrans,im_proj,NoDataValue]=read_rasterfile_setnan(I_Old_top_down_file)
	I_Old_sub_down=read_rasterfile_setnan(I_Old_sub_down_file)[0]
	k_ab_npp=read_rasterfile_setnan(k_ab_npp_file)[0]
	k_Old_top=read_rasterfile_setnan(k_Old_top_file)[0]
	k_Old_sub=read_rasterfile_setnan(k_Old_sub_file)[0]
	e_ab_npp_Old=read_rasterfile_setnan(e_ab_npp_Old_file)[0]
	fBNPP_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[0])[0]
	f20_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[1])[0]
	f20_100_im_data=read_rasterfile_setnan(fBNPP_f20_f20_100_raster_file_list[2])[0]
	all_NPP_file=get_all_type_files(NPP_time_series_filepath,'.tif')
	all_NPP_series_array=np.full([len(all_NPP_file),im_height,im_width],np.nan)
	all_Temp_file=get_all_type_files(Temp_series_filepath,'.tif')
	all_Temp_series_array=np.full([len(all_Temp_file),im_height,im_width],np.nan)
	all_SM_top_file=get_all_type_files(SM_top_series_filepath,'.tif')
	all_SM_top_series_array=np.full([len(all_SM_top_file),im_height,im_width],np.nan)
	all_SM_sub_file=get_all_type_files(SM_sub_series_filepath,'.tif')
	all_SM_sub_series_array=np.full([len(all_SM_sub_file),im_height,im_width],np.nan)
	print('loading NPP Temp SM_top SM_sub data')
	for year in range(len(all_NPP_file)):
		all_NPP_series_array[year]=read_rasterfile_setnan(all_NPP_file[year])[0]
		all_Temp_series_array[year]=read_rasterfile_setnan(all_Temp_file[year])[0]
		all_SM_top_series_array[year]=read_rasterfile_setnan(all_SM_top_file[year])[0]
		all_SM_sub_series_array[year]=read_rasterfile_setnan(all_SM_sub_file[year])[0]
	ft_function_name=['Century1','Century2','Daycent1','Daycent2','Q10_2','Q10_14','Lloyd_Taylor','Kirschbaum','Demeter','Standcarb']
	fw_function_name=['Century','Daycent','Demeter','Standcarb','Candy','Gompertz','Myers','Moyano','Skopp']
	first_year_NPP=read_rasterfile_setnan(all_NPP_file[0])[0]
	C_Young_Old_top_all_series_array=np.full([len(all_NPP_file),im_height,im_width],-999.0)
	C_Young_Old_sub_all_series_array=np.full([len(all_NPP_file),im_height,im_width],-999.0)
	x,y=np.where((~np.isnan(all_NPP_series_array[0])) & (~np.isnan(all_Temp_series_array[0]))& (~np.isnan(all_SM_top_series_array[0]))& (~np.isnan(all_SM_sub_series_array[0])))
	for k in range(x.shape[0]):
		i,j=x[k],y[k]  
		parameters_list=[I_Old_top_down[i,j], I_Old_sub_down[i,j], k_ab_npp[i,j], k_Old_top[i,j], k_Old_sub[i,j], e_ab_npp_Old[i,j]]
		first_year_C_Young_top_dynamics,first_year_C_Old_top_dynamics,first_year_C_Young_Old_top_all,first_year_C_Young_sub_dynamics,first_year_C_Old_sub_dynamics,first_year_C_Young_Old_sub_all=SoilC_Dynamics_4pools_to_optimize(400,first_year_NPP[i,j],fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j],I_Old_top_down[i,j], I_Old_sub_down[i,j], k_ab_npp[i,j], k_Old_top[i,j], k_Old_sub[i,j], e_ab_npp_Old[i,j])
		initial_carbon_pool_list=[first_year_C_Young_top_dynamics[-1],first_year_C_Old_top_dynamics[-1],first_year_C_Young_sub_dynamics[-1],first_year_C_Old_sub_dynamics[-1]]
		NPP_series=all_NPP_series_array[:,i,j]
		Temp_series=all_Temp_series_array[:,i,j]
		SM_top_series=all_SM_top_series_array[:,i,j]
		SM_sub_series=all_SM_sub_series_array[:,i,j]
		if np.any(np.isnan(NPP_series)) or np.any(np.isnan(Temp_series)) or np.any(np.isnan(SM_top_series)) or np.any(np.isnan(SM_sub_series)):
			pass
		else:
			C_Young_top_dynamics,C_Old_top_dynamics,C_Young_Old_top_all,C_Young_sub_dynamics,C_Old_sub_dynamics,C_Young_Old_sub_all=SoilC_Dynamics_4pools_for_simulation(NPP_series,Temp_series,SM_top_series, SM_sub_series, fBNPP_im_data[i,j],f20_im_data[i,j],f20_100_im_data[i,j],parameters_list, initial_carbon_pool_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario)
			C_Young_Old_top_all_series_array[:,i,j]=np.array(C_Young_Old_top_all)
			C_Young_Old_sub_all_series_array[:,i,j]=np.array(C_Young_Old_sub_all)
		print('Progress: '+ f"{round((k+1)*100.0/x.shape[0],4):.4f}"+'%',end='\r')
	for year in range(C_Young_Old_top_all_series_array.shape[0]):
		if K_driver_scenario==1:
			SOC20_filepath=Output_SOC_series_filepath+ft_function_name[fT_K_ratio_method_index]+'/SOC20/QL_SOC20_'+str(year+1982)+'_'+ft_function_name[fT_K_ratio_method_index]+'.tif'
			SOC20_100_filepath=Output_SOC_series_filepath+ft_function_name[fT_K_ratio_method_index]+'/SOC20_100/QL_SOC20_100_'+str(year+1982)+'_'+ft_function_name[fT_K_ratio_method_index]+'.tif'
			write_rasterfile(SOC20_filepath,C_Young_Old_top_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
			write_rasterfile(SOC20_100_filepath,C_Young_Old_sub_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
		elif K_driver_scenario==2:
			SOC20_filepath=Output_SOC_series_filepath+fw_function_name[fW_K_ratio_method_index]+'/SOC20/QL_SOC20_'+str(year+1982)+'_'+fw_function_name[fW_K_ratio_method_index]+'.tif'
			SOC20_100_filepath=Output_SOC_series_filepath+fw_function_name[fW_K_ratio_method_index]+'/SOC20_100/QL_SOC20_100_'+str(year+1982)+'_'+fw_function_name[fW_K_ratio_method_index]+'.tif'
			write_rasterfile(SOC20_filepath,C_Young_Old_top_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
			write_rasterfile(SOC20_100_filepath,C_Young_Old_sub_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
		elif K_driver_scenario==3:			
			SOC20_filepath=Output_SOC_series_filepath+ft_function_name[fT_K_ratio_method_index]+'/'+fw_function_name[fW_K_ratio_method_index]+'/SOC20/QL_SOC20_'+str(year+1982)+'_'+ft_function_name[fT_K_ratio_method_index]+'_'+fw_function_name[fW_K_ratio_method_index]+'.tif'
			SOC20_100_filepath=Output_SOC_series_filepath+ft_function_name[fT_K_ratio_method_index]+'/'+fw_function_name[fW_K_ratio_method_index]+'/SOC20_100/QL_SOC20_100_'+str(year+1982)+'_'+ft_function_name[fT_K_ratio_method_index]+'_'+fw_function_name[fW_K_ratio_method_index]+'.tif'
			write_rasterfile(SOC20_filepath,C_Young_Old_top_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
			write_rasterfile(SOC20_100_filepath,C_Young_Old_sub_all_series_array[year],im_width,im_height,1,im_geotrans,im_proj,-999.0)
	print('All raster saved')

if __name__ == '__main__':
	#1. Using PSO algorithm for spatial parameter calibration
	NPP_raster_file=u"model_input/NPP/Qinling_NPP_1982.tif"
	SOC_20_raster_file=u'model_input/SOC/QL_SOC20_1980s.tif'
	SOC_20_100_raster_file=u'model_input/SOC/QL_SOC20_100_1980s.tif'
	fBNPP_chn=u'model_input/ROOT_NPP_Parameter/fBNPP_QL.tif'
	f20cm_chn=u'model_input/ROOT_NPP_Parameter/f20cm_QL.tif'
	f20_100cm_chn=u'model_input/ROOT_NPP_Parameter/f20_100cm_QL.tif'
	spin_up_year=1000
	opti_para_all_data_txt_filepath=u'model_output/opti_para_alldata_txt_QL/'
	fBNPP_f20_f20_100_raster_file_list=[fBNPP_chn,f20cm_chn,f20_100cm_chn]
	start_line,end_line=0,592
	para_optimize_for_spatial_array_PSO(NPP_raster_file,SOC_20_raster_file,SOC_20_100_raster_file,fBNPP_f20_f20_100_raster_file_list,spin_up_year,opti_para_all_data_txt_filepath,start_line,end_line)

	#2. Convert parameter optimization results into raster files
	opti_para_all_data_txt_filepath=u'model_output/opti_para_alldata_txt_QL/'
	reference_raster_filepath=u"model_input/NPP/Qinling_NPP_1982.tif"
	output_raster_path=u'model_output/Results_QL/'
	read_txt_convert_raster(opti_para_all_data_txt_filepath,reference_raster_filepath,output_raster_path)

	#3. Conduct time series simulation
	NPP_time_series_filepath=u'model_input/NPP/'
	Temp_series_filepath=u'model_input/Temp/'
	fBNPP_chn=u'model_input/ROOT_NPP_Parameter/fBNPP_QL.tif'
	f20cm_chn=u'model_input/ROOT_NPP_Parameter/f20cm_QL.tif'
	f20_100cm_chn=u'model_input/ROOT_NPP_Parameter/f20_100cm_QL.tif'

	SM_top_series_filepath=u'model_input/SM_top/'
	SM_sub_series_filepath=u'model_input/SM_root/'

	fBNPP_f20_f20_100_raster_file_list=[fBNPP_chn,f20cm_chn,f20_100cm_chn]
	parameter_filepath=u'model_output/parameters/'

	#Simulation scenario: K_driver_Scenario=1, 2, 3 respectively represent only the fT and fW functions, as well as the combination of fT and fW functions
	K_driver_scenario=1

	if K_driver_scenario==1:
		Output_SOC_series_filepath=u'model_output/0_SOC_series_CASA_NPP_adjust_fT_Sierra_only/'
		simulate_SOC_based_on_NPP_Temp_SM(NPP_time_series_filepath,parameter_filepath,Temp_series_filepath,SM_top_series_filepath, SM_sub_series_filepath, fBNPP_f20_f20_100_raster_file_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario,Output_SOC_series_filepath)
	elif K_driver_scenario==2:
		Output_SOC_series_filepath=u'model_output/1_SOC_series_CASA_NPP_fW_Sierra_only/'
		simulate_SOC_based_on_NPP_Temp_SM(NPP_time_series_filepath,parameter_filepath,Temp_series_filepath,SM_top_series_filepath, SM_sub_series_filepath, fBNPP_f20_f20_100_raster_file_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario,Output_SOC_series_filepath)
	else:
		Output_SOC_series_filepath=u'model_output/3_SOC_series_CASA_NPP_fT_fW_Sierra_both/'
		for fT_K_ratio_method_index in range(10):
			for fW_K_ratio_method_index in range(9):
				simulate_SOC_based_on_NPP_Temp_SM(NPP_time_series_filepath,parameter_filepath,Temp_series_filepath,SM_top_series_filepath, SM_sub_series_filepath, fBNPP_f20_f20_100_raster_file_list,fT_K_ratio_method_index,fW_K_ratio_method_index,K_driver_scenario,Output_SOC_series_filepath)
				print('fW_model: '+str(fW_K_ratio_method_index+1)+' done')