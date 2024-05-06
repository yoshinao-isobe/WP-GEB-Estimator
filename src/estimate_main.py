# Copyright (C) 2024
# National Institute of Advanced Industrial Science and Technology (AIST)

# estimating generalization error upper bounds with perturbations on wights

import time
from absl import app
from absl import flags
import math

import estimate_params as prm
import utils as utl

EPS_ERR = 1e-10
NA = 'N/A'
Results_dir = 'results'


def main(args):

    # ------------------------------------
    # set parameters
    # ------------------------------------

    params = prm.OptParams(flags.FLAGS)
    root_result_dir = Results_dir + '/' + params.result_dir
    utl.chk_mkdir(root_result_dir)

    search_out_file = root_result_dir + '/' + params.search_file + '_out.csv'
    estimate_info_file = root_result_dir + '/' + params.estimate_file + '_info.txt'
    estimate_out_file = root_result_dir + '/' + params.estimate_file + '_out.csv'

    options = params.model_params()
    utl.save_message(estimate_info_file, options, 'w')
    print(options, end='')

    # ------------------------------------
    #  initialize bound_out
    # ------------------------------------

    search_out_list = utl.load_list(search_out_file)
    search_out_head = search_out_list[0]
    search_out_list.pop(0)

    if not utl.check_exist(estimate_out_file):
        # csv bound

        head_str = utl.list_to_str(search_out_head, ',') + ','
        head_str += 'gen_err_wst_adapt_ub,test_err_wst_adapt_ub,'
        head_str += 'err_thr_adapt_ub,err_thr_adapt,'
        head_str += 'conf_wst_adapt,conf0_wst_adapt,'
        head_str += 'gen_err_wst_fix_ub,test_err_wst_fix_ub,'
        head_str += 'err_thr_fix,conf_wst_fix,conf0_wst_fix,'
        head_str += 'gen_err_rnd_ub,test_err_rnd_ub,conf_rnd,conf0_rnd\n'

        utl.save_message(estimate_out_file, head_str, 'w')
        idx_st = 0

    else:
        estimate_out_dict_list = utl.load_csv_dict_list(estimate_out_file)
        idx_st = len(estimate_out_dict_list)

    # ----------------------------------------------
    #  load rnd_out
    # ----------------------------------------------

    search_out_dict_list = utl.load_csv_dict_list(search_out_file)
    search_out_size = len(search_out_dict_list)

    for idx in range(idx_st, search_out_size):

        avl_wst_adapt = True
        # avl_wst_fix = True
        # avl_rnd = True

        search_out_dict = search_out_dict_list[idx]

        datasize = int(search_out_dict['dataset_size'])

        perturb_ratio = float(search_out_dict['perturb_ratio'])
        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(estimate_info_file, info_str + '\n', 'a')
        print(info_str)

        perturb_sample_size = int(search_out_dict['perturb_sample_size'])

        # -------------------------------------
        # generalization error bound
        # -------------------------------------

        time1 = time.time()

        delta = params.delta
        delta0_ratio = params.delta0_ratio
        delta0 = delta * delta0_ratio

        # ----------------------------------------
        # worst perturbation (adaptive threshold)
        # ----------------------------------------

        err_num_str = search_out_dict['err_num']

        if err_num_str == NA:
            avl_wst_adapt = False
            err_thr_adapt = 0
            err_thr_adapt_ub = 0
            gen_err_wst_adapt_ub = 0
            test_err_wst_adapt_ub = 0
            conf_wst_adapt = 0
            conf0_wst_adapt = 0
        else:
            err_num = int(err_num_str)

            if datasize == err_num:
                test_err_wst_adapt_ub = 1.0
                gen_err_wst_adapt_ub = 1.0
                conf0_wst_adapt = 1.0
                conf_wst_adapt = 1.0
                err_thr_adapt = 0
                err_thr_adapt_ub = 0

            else:  # zero_err_num > 0
                test_err_wst_adapt_ub = err_num / datasize

                delta_ge = delta - delta0
                conf_wst_adapt = 1.0 - delta
                kl_ub = math.log(2.0 / delta_ge) / datasize
                gen_err_wst_adapt_ub = utl.inv_binary_kl_div(test_err_wst_adapt_ub, kl_ub, params.eps_nm, params.max_nm)

                # adaptive threshold upper bound
                zero_err_num = datasize - err_num  # S2
                conf0_wst_adapt = 1.0 - delta0

                delta_rnd1 = delta0 / zero_err_num
                err_thr1 = err_thr_wst(delta_rnd1, perturb_sample_size)
                err_thr_adapt = err_thr1 * zero_err_num / datasize

                kl_ub = math.log(2.0 / delta) / datasize
                zero_err_rate = zero_err_num / datasize
                zero_err_rate_ub = utl.inv_binary_kl_div(zero_err_rate, kl_ub, params.eps_nm, params.max_nm)
                zero_err_num_ub = zero_err_rate_ub * datasize
                delta_rnd1_ub = delta0 / zero_err_num_ub
                err_thr_fix_ub = err_thr_wst(delta_rnd1_ub, perturb_sample_size)
                err_thr_adapt_ub = zero_err_rate_ub * err_thr_fix_ub

                err_thr_max = err_thr_wst(delta0 / datasize, perturb_sample_size)
                if err_thr_adapt_ub > err_thr_max:
                    err_thr_adapt_ub = err_thr_max

        # ----------------------------------------
        # worst perturbation (fix threshold)
        # ----------------------------------------

        test_err_wst_fix_ub = float(search_out_dict['test_err_wst'])

        delta_ge = delta - delta0
        conf_wst_fix = 1.0 - delta
        kl_ub = math.log(2.0 / delta_ge) / datasize
        gen_err_wst_fix_ub = utl.inv_binary_kl_div(test_err_wst_fix_ub, kl_ub, params.eps_nm, params.max_nm)

        # fix threshold upper bound
        conf0_wst_fix = 1.0 - delta0
        delta_rnd1 = delta0 / datasize
        err_thr_fix = err_thr_wst(delta_rnd1, perturb_sample_size)

        # ----------------------------------------
        # random perturbation
        # ----------------------------------------

        test_err_rnd = float(search_out_dict['test_err_avr'])
        delta = params.delta

        if perturb_ratio == 0:
            delta_ge = delta
            conf0_rnd = 1
            # by Chernoff bound
            kl_ub = math.log(2.0 / delta_ge) / datasize
            gen_err_rnd_ub = utl.inv_binary_kl_div(test_err_rnd, kl_ub, params.eps_nm, params.max_nm)
            conf_rnd = 1.0 - delta
            test_err_rnd_ub = test_err_rnd

        else:
            delta0_ratio = params.delta0_ratio
            delta0 = delta * delta0_ratio
            conf0_rnd = 1 - delta0

            # by Perez-Ortiz 2021
            kl_ub_rnd = math.log(2.0 / delta0) / perturb_sample_size
            test_err_rnd_ub = utl.inv_binary_kl_div(test_err_rnd, kl_ub_rnd, params.eps_nm, params.max_nm)

            # upper bound of the generalization error
            delta_ge = delta - delta0
            # by Maurer bound 2004
            kl_ub = math.log(2 * math.sqrt(datasize) / delta_ge) / datasize
            gen_err_rnd_ub = utl.inv_binary_kl_div(test_err_rnd_ub, kl_ub, params.eps_nm, params.max_nm)
            conf_rnd = 1.0 - delta

        time2 = time.time()
        e_time = time2 - time1

        #
        info_str = ''

        if perturb_ratio == 0:
            info_str += '  No weight-perturbation:\n'
            info_str += '    Test error: '
            info_str += '{:.2f}%\n'.format(test_err_rnd_ub * 100)
        else:
            p_size = int(search_out_dict['perturb_sample_size'])
            info_str += '  Random perturbation sample size: {:d}\n'.format(p_size)

            # gen-ub (wst adapt)
            if avl_wst_adapt:
                info_str += '  Worst weight-perturbation (adaptive threshold):\n'
                info_str += '    Perturbed generalization error bound: '
                info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                    gen_err_wst_adapt_ub * 100, conf_wst_adapt * 100)
                info_str += '    Perturbed Test error bound: '
                info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                    test_err_wst_adapt_ub * 100, conf0_wst_adapt * 100)
                info_str += '    Adaptive threshold bound (expected): '
                info_str += '{:.4f}% (Conf: {:.2f}%)\n'.format(
                    err_thr_adapt_ub * 100, conf_wst_adapt * 100)
                info_str += '    Adaptive threshold (average): '
                info_str += '{:.4f}%\n'.format(err_thr_adapt * 100)

            # gen-ub (wst fix)
            info_str += '  Worst weight-perturbation (fixed threshold):\n'
            info_str += '    Perturbed generalization error bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                gen_err_wst_fix_ub * 100, conf_wst_fix * 100)
            info_str += '    Perturbed Test error bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                test_err_wst_fix_ub * 100, conf0_wst_fix * 100)
            info_str += '    Fixed threshold: '
            info_str += '{:.4f}%\n'.format(err_thr_fix * 100)

            # gen-ub (rnd)
            info_str += '  Random weight-perturbation:\n'
            info_str += '    Perturbed generalization error bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                gen_err_rnd_ub * 100, conf_rnd * 100)
            info_str += '    Perturbed Test error bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                test_err_rnd_ub * 100, conf0_rnd * 100)

        info_str += '  (Elapsed Time: {:.1f} [sec])\n'.format(e_time)

        utl.save_message(estimate_info_file, info_str + '\n', 'a')
        print(info_str)

        rnd_out = search_out_list[idx]
        val_csv = utl.list_to_str(rnd_out, ', ') + ', '

        if avl_wst_adapt:
            val_csv += str(gen_err_wst_adapt_ub) + ', '
            val_csv += str(test_err_wst_adapt_ub) + ', '
            val_csv += str(err_thr_adapt_ub) + ', '
            val_csv += str(err_thr_adapt) + ', '
            val_csv += str(conf_wst_adapt) + ', '
            val_csv += str(conf0_wst_adapt) + ', '
        else:
            val_csv += NA + ', ' + NA + ', ' + NA + ', ' + NA + ', ' + NA + ', ' + NA + ', '

        val_csv += str(gen_err_wst_fix_ub) + ', '
        val_csv += str(test_err_wst_fix_ub) + ', '
        val_csv += str(err_thr_fix) + ', '
        val_csv += str(conf_wst_fix) + ', '
        val_csv += str(conf0_wst_fix) + ', '

        val_csv += str(gen_err_rnd_ub) + ', '
        val_csv += str(test_err_rnd_ub) + ', '
        val_csv += str(conf_rnd) + ', '
        val_csv += str(conf0_rnd) + '\n'

        utl.save_message(estimate_out_file, val_csv, 'a')

    return


# threshold for worst case
def err_thr_wst(delta1, perturb_size):
    if delta1 >= 2.0:
        err_thr = 0.0
    else:
        err_thr = 1.0 - math.pow(delta1 / 2.0, 1 / perturb_size)
    return err_thr


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
