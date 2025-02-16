# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# estimating generalization risk/error upper bounds with perturbations on weights

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

    measure_out_file = root_result_dir + '/' + params.measure_file + '_out.csv'
    estimate_info_file = root_result_dir + '/' + params.estimate_file + '_info.txt'
    estimate_out_file = root_result_dir + '/' + params.estimate_file + '_out.csv'

    top_line = '\n------------------------------\n'
    options = top_line + params.model_params()
    utl.save_message(estimate_info_file, options, 'a')
    print(options, end='')

    # ------------------------------------
    #  initialize bound_out
    # ------------------------------------

    measure_out_list = utl.load_list(measure_out_file)
    measure_out_head = measure_out_list[0]
    measure_out_list.pop(0)

    if not utl.check_exist(estimate_out_file):
        # csv bound

        head_str = utl.list_to_str(measure_out_head, ',') + ','
        head_str += 'gen_risk_ub,test_risk_ub,conf_risk,conf0_risk,'
        head_str += 'non_det_rate_ub,gen_err_thr_ub,'
        head_str += 'gen_err_ub,test_err_ub,test_err,conf_err,conf0_err\n'

        utl.save_message(estimate_out_file, head_str, 'w')
        idx_st = 0

    else:
        estimate_out_dict_list = utl.load_csv_dict_list(estimate_out_file)
        idx_st = len(estimate_out_dict_list)

    # ----------------------------------------------
    #  load err_out
    # ----------------------------------------------

    measure_out_dict_list = utl.load_csv_dict_list(measure_out_file)
    measure_out_size = len(measure_out_dict_list)

    for idx in range(idx_st, measure_out_size):

        # avl_risk_fix = True

        measure_out_dict = measure_out_dict_list[idx]

        datasize = int(measure_out_dict['dataset_size'])

        perturb_ratio = float(measure_out_dict['perturb_ratio'])
        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(estimate_info_file, info_str + '\n', 'a')
        print(info_str)

        perturb_sample_size = int(measure_out_dict['perturb_sample_size'])
        err_num_search = int(measure_out_dict['err_num_search'])
        err_thr = float(measure_out_dict['err_thr'])

        delta = float(measure_out_dict['delta'])
        delta0_ratio = float(measure_out_dict['delta0_ratio'])
        delta0 = delta * delta0_ratio

        # -------------------------------------
        # generalization risk/error bound
        # -------------------------------------

        time1 = time.time()

        # ----------------------------------------
        # generalization risk bound
        # ----------------------------------------

        err_num_str = measure_out_dict['err_num']

        err_num = int(err_num_str)

        if datasize == err_num:
            # test_risk = 1.0
            test_risk_ub = 1.0
            gen_risk_ub = 1.0
            gen_err_thr_ub = 0
            conf_risk = 1.0
            conf0_risk = 1.0
            non_det_rate_ub = 0

        elif perturb_ratio == 0:
            test_err = float(measure_out_dict['test_err_avr'])
            delta_ge = delta
            conf0_risk = 1
            # by Chernoff bound
            kl_ub = math.log(1.0 / delta_ge) / datasize
            gen_risk_ub = utl.inv_binary_kl_div(test_err, kl_ub, params.eps_nm, params.max_nm)
            conf_risk = 1.0 - delta
            test_risk_ub = test_err

            gen_err_thr_ub = 0
            non_det_rate_ub = 1

        else:  # err_num < datasize

            # upper bound of generalized acceptable threshold
            non_det_rate = 1 - err_num_search / datasize
            kl_ub = math.log(1.0 / delta) / datasize
            non_det_rate_ub = utl.inv_binary_kl_div(non_det_rate, kl_ub, params.eps_nm, params.max_nm)
            gen_err_thr_ub = err_thr * non_det_rate_ub

            # upper bound of test risk rate
            test_risk_ub = err_num / datasize
            delta_ge = delta - delta0
            kl_ub = math.log(1.0 / delta_ge) / datasize
            gen_risk_ub = utl.inv_binary_kl_div(test_risk_ub, kl_ub, params.eps_nm, params.max_nm)

            # confidence
            conf_risk = 1.0 - delta
            conf0_risk = 1.0 - delta0

        # ----------------------------------------
        # generalization error bound
        # ----------------------------------------

        # search has been skipped
        if err_num_search == 0:
            avl_err = True
            test_err = float(measure_out_dict['test_err_avr'])

            if perturb_ratio == 0:
                delta_ge = delta
                conf0_err = 1
                # by Chernoff bound
                # kl_ub = math.log(1.0 / delta_ge) / datasize
                # gen_err_ub = utl.inv_binary_kl_div(test_err, kl_ub, params.eps_nm, params.max_nm)
                gen_err_ub = gen_risk_ub
                conf_err = 1.0 - delta
                test_err_ub = test_err

            else:
                # by Perez-Ortiz 2021
                kl_ub_err = math.log(1.0 / delta0) / perturb_sample_size
                test_err_ub = utl.inv_binary_kl_div(test_err, kl_ub_err, params.eps_nm, params.max_nm)

                # upper bound of the generalization error
                delta_ge = delta - delta0
                # by Maurer bound 2004
                kl_ub = math.log(2 * math.sqrt(datasize) / delta_ge) / datasize
                gen_err_ub = utl.inv_binary_kl_div(test_err_ub, kl_ub, params.eps_nm, params.max_nm)

                conf0_err = 1 - delta0
                conf_err = 1 - delta

        # random perturbation is not available.
        else:
            avl_err = False

            test_err = 0
            test_err_ub = 0
            gen_err_ub = 0
            conf0_err = 0
            conf_err = 0

        time2 = time.time()
        e_time = time2 - time1

        # -----------------------------
        # save and print results
        # -----------------------------

        info_str = ''

        if perturb_ratio == 0:
            info_str += '  No weight-perturbation:\n'
            info_str += '    Generalization error bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                gen_err_ub * 100, conf_err * 100)
            info_str += '    Test error: '
            info_str += '{:.2f}%\n'.format(
                test_err * 100)

        else:
            p_size = int(measure_out_dict['perturb_sample_size'])
            info_str += '  Random perturbation sample size: {:d}\n'.format(p_size)

            # gen-ub (risk adapt)
            if err_num_search == 0:
                info_str += '  Risk (without search):\n'
            else:
                info_str += '  Risk (with search):\n'
            info_str += '    Perturbed generalization risk bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                gen_risk_ub * 100, conf_risk * 100)
            info_str += '    Perturbed test risk bound: '
            info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                test_risk_ub * 100, conf0_risk * 100)
            info_str += '    Generalization acceptable threshold bound: '
            info_str += '{:.4f}% (Conf: {:.2f}%)\n'.format(
                gen_err_thr_ub * 100, conf_risk * 100)

            if avl_err:
                # gen-ub (err)
                info_str += '  Error:\n'
                info_str += '    Perturbed generalization error bound: '
                info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                    gen_err_ub * 100, conf_err * 100)
                info_str += '    Perturbed test error bound: '
                info_str += '{:.2f}% (Conf: {:.2f}%)\n'.format(
                    test_err_ub * 100, conf0_err * 100)

        info_str += '  (Elapsed Time: {:.1f} [sec])\n'.format(e_time)

        utl.save_message(estimate_info_file, info_str + '\n', 'a')
        print(info_str)

        msr_out = measure_out_list[idx]
        val_csv = utl.list_to_str(msr_out, ', ') + ', '

        val_csv += str(gen_risk_ub) + ', '
        val_csv += str(test_risk_ub) + ', '
        val_csv += str(conf_risk) + ', '
        val_csv += str(conf0_risk) + ', '
        val_csv += str(non_det_rate_ub) + ', '
        val_csv += str(gen_err_thr_ub) + ', '

        if avl_err:
            val_csv += str(gen_err_ub) + ', '
            val_csv += str(test_err_ub) + ', '
            val_csv += str(test_err) + ', '
            val_csv += str(conf_err) + ', '
            val_csv += str(conf0_err) + '\n'
        else:
            val_csv += NA + ', ' + NA + ', ' + NA + ', ' + NA + ', ' + NA + '\n'

        utl.save_message(estimate_out_file, val_csv, 'a')

    return


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
