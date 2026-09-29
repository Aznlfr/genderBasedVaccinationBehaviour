
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, interpolate
from scipy.ndimage.interpolation import shift
import time
import numpy.random as random
import multiprocessing
from multiprocessing import Pool
import sys
import os
#  #################################Saving Directory###########################################
dirname = os.path.dirname(__file__)
mainPath = os.path.join(dirname, 'USA/')
dataPath = 'Data/'
resultPath = 'Result/'
#mainPath = "~/"
script_name = os.path.basename(__file__)
inp = sys.argv
#usage = ("Usage: python ensemble.py ITERATIONS [FIT_SIGMA] [SEPARATE_CV] "
 #        "[SEPARATE_CVBAR] [SEPARATE_ALPHA]\n"
  #       "Each switch is T or F. Defaults: F T F T.\n"
   #      "FIT_SIGMA: T fits sigma in [0, 1]; F fixes sigma at 1.\n"
    #     "SEPARATE_*: T fits two parameters; F fits one shared parameter.")
fit_sigma = inp[2] == 'T' if len(inp) > 2 else True
separate_cv = inp[3] == 'T' if len(inp) > 3 else True
separate_cvbar = inp[4] == 'T' if len(inp) > 4 else True
separate_alpha = inp[5] == 'T' if len(inp) > 5 else True
iter_interval = int(inp[1]) # Number of iteration
max_ci = inp[6] == 'T' if len(inp) > 6 else True
ntrain = int(inp[7]) if len(inp) > 7 else 0
result_name = (f"{Path(script_name).stem}_sigma{'T' if fit_sigma else 'F'}"
               f"_cv{'T' if separate_cv else 'F'}"
               f"_cvbar{'T' if separate_cvbar else 'F'}"
               f"_alpha{'T' if separate_alpha else 'F'}"
               f"_smallci{'T' if max_ci else 'F'}"
               f"_train{'T' if ntrain>0 else 'F'}"
)

epd_file = mainPath + dataPath + 'EpiVacFile_6_week_added.csv'
Fit_pow = mainPath + dataPath + 'fit_best_side_effect.csv'
hesitancyFile = mainPath + dataPath + 'strongly_hesitant_average.csv' # proportion of strongly hesitant, vaccine refusers
vaxdata = mainPath + dataPath + 'vax_gender_12_week_added.csv' # 1 male, 2 female, 3 sex_unknown
pop = mainPath + dataPath + 'subpopVacSex_1.csv'
##################################Reading Directory#########################################
fitPow = pd.read_csv(Fit_pow)
hesitancy = pd.read_csv(hesitancyFile)
vaxSex = pd.read_csv(vaxdata)
popSex = pd.read_csv(pop)
OWID = pd.read_csv(epd_file)
vaxSex = vaxSex.reset_index()
OWID = OWID.reset_index()
OWID['Date'] = pd.to_datetime(OWID['Date'])
vaxSex['Date'] = pd.to_datetime(vaxSex['Date'])
OWID.sort_values(by='Date', inplace=True)
vaxSex.sort_values(by='Date', inplace=True)
method0 = 'LSODA'
StateName = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS',
             'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
             'OH', 'OK', 'OR', 'PA',  'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

def unpack_parameters(parameters):
    values = iter(parameters)
    alpha_1 = next(values)
    alpha_2 = next(values) if separate_alpha else alpha_1
    cv_m = next(values)
    cv_w = next(values) if separate_cv else cv_m
    cvbar_m = next(values)
    cvbar_w = next(values) if separate_cvbar else cvbar_m
    tilde_c_i = next(values)
    rate1 = next(values)
    sigma = next(values) if fit_sigma else 1.0
    return alpha_1, alpha_2, cv_m, cv_w, cvbar_m, cvbar_w, tilde_c_i, rate1, sigma


def system_dynamics(x, y, alpha_m, alpha_w, cv_m0, cv_w0, cbar_m, cbar_w, c_i, k1, sigma, *arg_N):
    leng = 11
    lenn, N12, h_m, h_w, Nm, Nw,  cvcof_m, cvcof_w, fun_mn, fun_wm, Ntot = arg_N[0:leng]
    time_ref = arg_N[leng:leng + lenn]
    number_of_case = arg_N[leng + lenn: leng + 2 * lenn]
    number_of_death = arg_N[leng + 2 * lenn: leng + 3 * lenn]
    frac_avail_dose = arg_N[leng + 3 * lenn:leng + 4 * lenn]
    number_of_case_interpolated = interpolate.interp1d(time_ref, number_of_case, kind='zero', axis=0, fill_value=
    "extrapolate")
    number_of_death_interpolated = interpolate.interp1d(time_ref, number_of_death, kind='zero', axis=0, fill_value=
    "extrapolate")
    frac_avail_dose_interpolated1 = interpolate.interp1d(time_ref, frac_avail_dose, kind='zero', axis=0, fill_value=
    "extrapolate")
    fr_Ne_m = (1.0 -h_m)*Nm/N12
    fr_Ne_w = (1.0 - h_w)*Nw/N12
    im_vax_m, br_vax_m, im_vax_w, br_vax_w = y
    totalvax_m = im_vax_m + br_vax_m  # proportion of men vaccinated to whole Nf12
    totalvax_w = im_vax_w + br_vax_w  # proportion of women vaccinated to whole Nf12
    frac_avail_dose_interpolated = frac_avail_dose_interpolated1(x) - (totalvax_m + totalvax_w)
    if frac_avail_dose_interpolated < 0.0:
        frac_avail_dose_interpolated = 0.0
    if fun_mn == 'p':
        cv_m = cv_m0 * (x-time_ref[0]+1)**cvcof_m
    else:
        cv_m = cv_m0 * np.exp(cvcof_m * (x - time_ref[0]))
    if fun_wm == 'p':
        cv_w = cv_w0*(x-time_ref[0]+1)**cvcof_w
    else:
        cv_w = cv_w0 * np.exp(cvcof_w * (x - time_ref[0]))
    epidm = c_i * (number_of_case_interpolated(x) / Ntot) +\
                      (number_of_death_interpolated(x) / Ntot)
    mv_mu = -cv_m + cbar_m + epidm
    wv_wu = -cv_w + cbar_w + epidm
    mv_wu = -cv_m + cbar_w + epidm
    wv_mu = -cv_w + cbar_m + epidm
    max_mv_mu = max(mv_mu, 0.0)
    max_wv_wu = max(wv_wu, 0.0)
    max_mv_wu = max(mv_wu, 0.0)
    max_wv_mu = max(wv_mu, 0.0)
    # Compute intermediate values
    intermediate_m = (1 - alpha_m) * fr_Ne_m - im_vax_m
    intermediate_w = (1 - alpha_w) * fr_Ne_w - im_vax_w
    intermediate_br_m = alpha_m * fr_Ne_m - br_vax_m
    intermediate_br_w = alpha_w * fr_Ne_w - br_vax_w

    # Calculate num_im_reg_m, num_im_reg_w, num_br_reg_m, and num_br_reg_w using the conditions
    conditions = {
        'num_im_reg_m': intermediate_m * sigma* (totalvax_m * max_mv_mu + totalvax_w * max_wv_mu) if
                                                                                intermediate_m > 0 else 0.0,
        'num_im_reg_w': intermediate_w * sigma* (totalvax_m * max_mv_wu + totalvax_w * max_wv_wu) if
                                                                                intermediate_w > 0 else 0.0,
        'num_br_reg_m': intermediate_br_m if mv_mu > 0.0 and intermediate_br_m > 0 else 0.0,
        'num_br_reg_w': intermediate_br_w if wv_wu > 0.0 and intermediate_br_w > 0 else 0.0
    }

    # Calculate total_reg
    total_reg = sum(conditions.values())
    # Calculate dot_num_im_vax_m, dot_num_br_vax_m, dot_num_im_vax_w, and dot_num_br_vax_w
    if total_reg > frac_avail_dose_interpolated:
        factor = frac_avail_dose_interpolated / total_reg
    else:
        factor = 1.0

    dot_num_im_vax_m = k1 * factor * conditions['num_im_reg_m']
    dot_num_br_vax_m = k1 * factor * conditions['num_br_reg_m']
    dot_num_im_vax_w = k1 * factor * conditions['num_im_reg_w']
    dot_num_br_vax_w = k1 * factor * conditions['num_br_reg_w']

    return dot_num_im_vax_m, dot_num_br_vax_m, dot_num_im_vax_w, dot_num_br_vax_w

def OBJ_ODE(parameters, *arg):

    lnght = 11
    ssize, Nf12, Hesitant_m, Hesitant_w, Nsex_m, Nsex_w, cv_cof_m, cv_cof_w, fun_m, fun_w, totalN = arg[0:lnght]
    time_ref = arg[lnght:lnght + ssize]
    new_vaccinated_00 = arg[lnght + ssize:lnght + 2 * ssize]
    new_vaccinated_01 = arg[lnght + 2 * ssize:lnght + 3 * ssize]
    Number_of_Case = arg[lnght + 3 * ssize:lnght + 4 * ssize]
    Number_of_Death = arg[lnght + 4 * ssize:lnght + 5 * ssize]
    Fraction_of_Eligible = arg[lnght + 5 * ssize:lnght + 6 * ssize]
    alpha_1, alpha_2, cv1_m, cv1_w, cv_barm, cv_barw, tilde_c_i_1, k1, sigma = unpack_parameters(parameters)
    arg_N = [len(time_ref), Nf12, Hesitant_m, Hesitant_w, Nsex_m, Nsex_w, cv_cof_m, cv_cof_w, fun_m, fun_w, totalN]

    sol = integrate.solve_ivp(system_dynamics, [time_ref[0], time_ref[-1]], (
                            0, 0, 0, 0),  max_step = 0.1, atol=1e-9, rtol=1e-7, args=(alpha_1, alpha_2, cv1_m, cv1_w, cv_barm, cv_barw, tilde_c_i_1,
                                                                             k1, sigma, *arg_N, *time_ref,
                                                                             *Number_of_Case, *Number_of_Death,
                                                                             *Fraction_of_Eligible),
                          t_eval=time_ref, dense_output=True, method=method0)
    model_output_0 = Nf12* sol.y[0, :] + Nf12*sol.y[1, :]
    model_output_shifted_0 = np.insert(model_output_0, 0, 0)
    model_output_shifted_0 = np.delete(model_output_shifted_0, [-1])
    ee_0 = np.asarray(model_output_0 - model_output_shifted_0)
    residuals_0 = ee_0 - np.asarray(new_vaccinated_00)
    model_output_1 = Nf12* sol.y[2, :] + Nf12*sol.y[3, :]
    model_output_shifted_1 = np.insert(model_output_1, 0, 0)
    model_output_shifted_1 = np.delete(model_output_shifted_1, [-1])
    ee_1 = np.asarray(model_output_1 - model_output_shifted_1)
    residuals_1 = ee_1 - np.asarray(new_vaccinated_01)
    RSE_22 = np.sum(residuals_0 ** 2) + np.sum(residuals_1**2)
    return RSE_22

def optm(arg): #

    i, seed_number = arg
    Ref_Time = [] #count_dose_1_female #count_dose_1_male
    total_vaccinated_0 = (vaxSex.loc[(vaxSex['Location'] == i), 'count_dose_1_male']).to_numpy(dtype=np.float64)
    total_shifted_0 = np.insert(total_vaccinated_0, 0, 0)
    total_shifted_0 = np.delete(total_shifted_0, [-1])
    new_vaccinated_0 = total_vaccinated_0 - total_shifted_0 #weekly number of vaccinated male
    total_vaccinated_1 = (vaxSex.loc[(vaxSex['Location'] == i),
                                     'count_dose_1_female']).to_numpy(dtype=np.float64)
    total_shifted_1 = np.insert(total_vaccinated_1, 0, 0)
    total_shifted_1 = np.delete(total_shifted_1, [-1])
    new_vaccinated_1 = total_vaccinated_1 - total_shifted_1 # weekly number of vaccinated female
    Number_of_Case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
    Number_of_Death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
    dose_available = OWID.loc[OWID['Location'] == i, 'dose_distributed'].to_numpy(dtype=np.float64)  # dose_distributed
    Nsex_m = popSex.loc[(popSex['Location'] == i) & (popSex['sex'] == 1), 'popf12'].iloc[0]
    Nsex_w = popSex.loc[(popSex['Location'] == i) & (popSex['sex'] == 2), 'popf12'].iloc[0]
    N = OWID.loc[OWID['Location'] == i, 'popf12'].iloc[0]
    Ntotal = OWID.loc[OWID['Location'] == i, 'poptotal'].iloc[0]  # total population
    cof_side_effect_m = fitPow.loc[(fitPow['Location'] == i) & (fitPow['sex'] == 1), 'parameter'].iloc[0]
    cof_side_effect_w = fitPow.loc[(fitPow['Location'] == i) & (fitPow['sex'] == 2), 'parameter'].iloc[0]
    fun_side_effect_m = fitPow.loc[(fitPow['Location'] == i) & (fitPow['sex'] == 1), 'minerror'].iloc[0]
    fun_side_effect_w = fitPow.loc[(fitPow['Location'] == i) & (fitPow['sex'] == 2), 'minerror'].iloc[0]
    #NonHesitant_m = Nsex_m*(1-hesitancy.loc[(hesitancy['Location'] == i) & (hesitancy['sex'] == 1),  'avg_proportion_strongly_not'].iloc[0])
    #NonHesitant_w = Nsex_w*(1-hesitancy.loc[(hesitancy['Location'] == i) & (hesitancy['sex'] == 2), 'avg_proportion_strongly_not'].iloc[0])
    Hesitant_m = (hesitancy.loc[(hesitancy['Location'] == i) & (hesitancy['sex'] == 1),  'avg_proportion_strongly_not'].iloc[0])
    Hesitant_w = (hesitancy.loc[(hesitancy['Location'] == i) & (hesitancy['sex'] == 2), 'avg_proportion_strongly_not'].iloc[0])
    fraction_dose_available = np.clip(dose_available / N, 0, None)
    death_case_fraction = Number_of_Death/Number_of_Case
    min_death_case =  min(death_case_fraction[(death_case_fraction>0) & (np.isfinite(death_case_fraction))])
    Ref_Time_2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
    for k in Ref_Time_2:
        delta = k - Ref_Time_2[0]
        Ref_Time.append(delta.days)
    Time_Ref = np.array(Ref_Time, dtype=np.float64)
    Time_Ref = np.divide(Time_Ref, 7.0)
    if ntrain >0:
       cutted_Time_Ref = Time_Ref[:-ntrain]
       cutted_new_vaccinated_0 = new_vaccinated_0[:-ntrain]
       cutted_new_vaccinated_1= new_vaccinated_1[:-ntrain]
       cutted_Number_of_Case = Number_of_Case[:-ntrain]
       cutted_Number_of_Death = Number_of_Death[:-ntrain]
       cutted_fraction_dose_available = fraction_dose_available[:-ntrain]
    else:
       cutted_Time_Ref = Time_Ref[:]
       cutted_new_vaccinated_0 = new_vaccinated_0[:]
       cutted_new_vaccinated_1= new_vaccinated_1[:]
       cutted_Number_of_Case = Number_of_Case[:]
       cutted_Number_of_Death = Number_of_Death[:]
       cutted_fraction_dose_available = fraction_dose_available[:]

################ TRY ############## alpha_1, alpha_2, cv1_m, cv1_w,  cv_bar01,  tilde_c_i_1,  k1
    max_rate = 10
    max_cbar0 = 1
    max_tilde = min_death_case if max_ci else 1.0
    np.random.seed(seed_number)
    cv_mg = random.random()
    cv_bar_g = max_cbar0*random.random()
    tilde_C_I_g =  max_tilde* random.random()
    rate_g1 = 0 + max_rate * random.random()
    cv_wg =  random.random()
    alpha_m_g = 0 + random.random()
    alpha_w_g = 0 + random.random()
    int_guess = [alpha_m_g]
    limit_list = [(0, 1)]
    if separate_alpha:
        int_guess.append(alpha_w_g)
        limit_list.append((0, 1))
    int_guess.append(cv_mg)
    limit_list.append((0, 1))
    if separate_cv:
        int_guess.append(cv_wg)
        limit_list.append((0, 1))
    int_guess.append(cv_bar_g)
    limit_list.append((0, max_cbar0))
    if separate_cvbar:
        int_guess.append(max_cbar0 * random.random())
        limit_list.append((0, max_cbar0))
    int_guess.extend([tilde_C_I_g, rate_g1])
    limit_list.extend([(0.0, max_tilde), (0, max_rate)])
    if fit_sigma:
        int_guess.append(random.random())
        limit_list.append((0, 1))
    int_guess = np.array(int_guess)
    startTime = time.time()  #ssize, N, NonHesitant_m, NonHesitant_w, cv_cof, Ntota
    res_opt = optimize.differential_evolution(OBJ_ODE, bounds=limit_list, polish=False, x0=int_guess, maxiter=
    iter_interval,  seed=random.default_rng(seed=seed_number),
                                              args=[len(cutted_Time_Ref), N, Hesitant_m, Hesitant_w, Nsex_m, Nsex_w,cof_side_effect_m,
                                                    cof_side_effect_w, fun_side_effect_m, fun_side_effect_w,
                                                    Ntotal,  *cutted_Time_Ref, *cutted_new_vaccinated_0,
                                                                                      *cutted_new_vaccinated_1,
                                                              *cutted_Number_of_Case, *cutted_Number_of_Death,
                                                    *cutted_fraction_dose_available])
# Extracting the estimated parameters ssize, Nf12, NonHesitant_m, NonHesitant_w, cv_cof_m, cv_cof_w, fun_m, fun_w, totalN
    alpha_1, alpha_2, cv_m, cv_w, cvbar_m, cvbar_w, tilde_c_i, rate1, sigma = unpack_parameters(res_opt.x)
    RSE_1 = res_opt.fun

    with open(mainPath + resultPath + result_name + '.txt', "a") as text_file:
            print(f"{i}, alpha_1: {alpha_1}, alpha_2: {alpha_2}, cv_m: {cv_m}, cv_w:{cv_w}, cvbar_m: {cvbar_m}, cvbar_w: {cvbar_w}, "
                  f"tilde_c_i: {tilde_c_i}, sigma: {sigma}, "
                  f"k1: {rate1},"
                  f" RSE: {RSE_1}, seed:{seed_number}, msg: {res_opt.message}, success: {res_opt.success}", file=text_file)

    return i, alpha_1, alpha_2, cv_m, cv_w, cvbar_m, cvbar_w, tilde_c_i, rate1, sigma, RSE_1, seed_number


def main():
    filepath = Path(mainPath + resultPath + result_name + '_estimated.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result_dataframe = pd.DataFrame(columns=['code', 'alpha_1', 'alpha_2', 'cv_m', 'cv_w', 'cvbar_m', 'cvbar_w',
                                             'tilde_c_i', 'k1', 'sigma',  'RSE_1', 'seed'])
    #Number_of_cpus = multiprocessing.cpu_count()
    Number_of_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    seed_pool = [2026]
    script_name = os.path.basename(__file__)
    for seed_run in seed_pool:
        seed_vec = [seed_run]*len(StateName)
        with Pool(processes=Number_of_cpus) as run_pool:
            parallel_output = run_pool.map(optm, zip(StateName, seed_vec))
            run_pool.close()
            run_pool.join()
        f = parallel_output
        ddf = pd.DataFrame(f, columns=['code', 'alpha_1', 'alpha_2', 'cv_m', 'cv_w', 'cvbar_m', 'cvbar_w' , 'tilde_c_i',
                                       'k1', 'sigma',  'RSE_1', 'seed'])
        result_dataframe = pd.concat([ddf, result_dataframe])
    result_dataframe.to_csv(filepath)


if __name__ == '__main__':
    main()

