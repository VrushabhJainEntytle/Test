#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dataframe_image as dfi
import matplotlib.font_manager
from rdp import rdp
import os
import json
import decimal
from util.generic_utils import check_package_versions, setup_pandas_display
import argparse
from dao import base_dao
import datetime
from os.path import abspath, join, dirname
from service_config.training_configuration import global_config
from input_data_preppers.transaction_history_segmentation_data_prepper import TransactionHistorySegmentationFeaturesQueryDescriptionPrepper
from dao.prediction_results_dao import save_segmentation_output
import logging


logging.getLogger('matplotlib.font_manager').disabled = True
pd.options.display.float_format = '{:.2f}'.format
cache_plot_directory = '/home/ec2-user/email-attachments'


def prep_inputdata_segmentation(engine_handle,data_caching):
    cross_query_args = {
        'client_id': global_config.client_id,
        'start_date': global_config.start_date,
        'end_date': global_config.end_date
    }

    transaction_history = TransactionHistorySegmentationFeaturesQueryDescriptionPrepper(cross_query_args,
                                                                                        engine_handle=engine_handle,
                                                                                        caching=data_caching)
    logging.info('Finished loading PE transaction data for segmentation')
    transaction_history_data = transaction_history.get_prepped_input_data()
    transaction_history_data = transaction_history_data[0]
    return transaction_history_data


def agg_rankingparam_by_customerlevelspecification(transaction_history,customer_level_specification):
    ranking_param = ['numberOfEquipmentTransactions','numberOfEquipments', 'equipmentRevenue','numberOfRelevantPartsTransactions',
                     'numberOfRelevantParts','relevantPartsRevenue', 'numberOfOtherPartsTransactions','numberOfOtherParts', 'otherPartsRevenue',
                     'numberOfTotalPartsTransactions', 'numberOfTotalParts','totalPartsRevenue', 'numberOfServiceTransactions', 'serviceRevenue',
                     'numberOfServiceContractTransactions', 'serviceContractRevenue']
    agg_transaction_history = transaction_history[(list(customer_level_specification) + ranking_param)]
    agg_transaction_history = agg_transaction_history.groupby(list(customer_level_specification)).sum().reset_index()
    agg_transaction_history = transaction_history[transaction_history.columns.drop(ranking_param)].merge(agg_transaction_history,
                                                                                                         how = 'inner',
                                                                                                         on = list(customer_level_specification))
    return agg_transaction_history


def prep_segmentation_data(agg_transaction_history,customer_level_specification, cohort_variable):
    required_features = list(dict.fromkeys(['purchasingEntityId'] + list(customer_level_specification) + [cohort_variable]))
    filtered_agg_transaction_history = agg_transaction_history[required_features]
    filtered_agg_transaction_history = filtered_agg_transaction_history.sort_values(cohort_variable,ascending = False)
    filtered_agg_transaction_history = filtered_agg_transaction_history.reset_index(drop = True)
    filtered_agg_transaction_history.insert(len(filtered_agg_transaction_history.columns),('log'+ cohort_variable),np.nan)
    filtered_agg_transaction_history.loc[filtered_agg_transaction_history[cohort_variable] > 0,'log'+ cohort_variable] = np.log10(filtered_agg_transaction_history.loc[filtered_agg_transaction_history[cohort_variable] > 0,cohort_variable])
    filtered_agg_transaction_history['rank'] = filtered_agg_transaction_history[cohort_variable].rank(method = 'dense',ascending = False)  # Assigning rank
    filtered_agg_transaction_history['rankPercentile'] = (filtered_agg_transaction_history[cohort_variable].rank(method = 'dense',pct = True))*100
    filtered_agg_transaction_history['logRank'] = np.log10(filtered_agg_transaction_history['rank']) # log of rank
    return filtered_agg_transaction_history


def compute_slope(x_var,y_var):
    slope = np.diff(y_var)/np.diff(x_var)
    return slope


def create_segment_at_epsilon(original_curve,min_slope_change,epsilon_value,min_top_segment_rank):
    # Build simplified (approximate) curve using RDP algorithm.
    simplified_curve = rdp(original_curve, epsilon=epsilon_value)
    #Store x and y co-ordinate of simplified curve
    sx, sy = simplified_curve.T
    # Compute the slope on the simplified_curve.
    slope = compute_slope(sx,sy)
    #Compute avg rate of change of slope
    avg_slope_change_rate = np.round(np.diff(slope),1)
    # Select the index of the points with avg rate of change of slope >= min_slope_change
    idx = np.where(np.absolute(avg_slope_change_rate) >= min_slope_change)[0] + 1
    y_var = pd.Series(original_curve.T[1])
    #exclude cut off points if falls among top min_top_segment_rank
    rank_cutoff = idx[[y_var[y_var==i].index[0]<= min_top_segment_rank for i in sy[idx]]]
    idx = np.setdiff1d(idx,rank_cutoff)
    y_cutoff_points = [10**i for i in sy[idx]]
    created_segments = len(idx) + 1
    return created_segments,idx,y_cutoff_points,sx,sy


def save_segment_plot(path,config,original_curve,sx,sy,idx,cohort_variable,segment_type):
    #%matplotlib widget
    fig = plt.figure(1, figsize=(8,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,3])
    ax1 = fig.add_subplot(gs[0])
    x_var = pd.Series(original_curve.T[0])
    y_var = pd.Series(original_curve.T[1])
    # Visualize original curve and its simplified version.
    ax1.plot(x_var, y_var, 'k+', label='original')
    ax1.plot(sx, sy, 'y-', label='simplified')
    ax1.plot(sx[idx], sy[idx], 'ro', markersize = 4, label='change points')
    #Display co-ordinate of cut off points on plot
    for i_x, i_y in zip(sx[idx], sy[idx]):
        ax1.text(i_x, i_y, '({}, {})'.format(np.round(i_x,3), np.round(i_y,3)))
    ax1.legend(loc='lower left')
    ax1.set_xlabel('Log(Rank)')
    ax1.set_ylabel(f'log{cohort_variable}')
    ax1.set_title(f'Rank vs {cohort_variable}')
    ax1.grid(True)
    plt.close()
    ax1.figure.savefig(abspath(join(path,
              "client_id_{}_{}_plot_{}_{}_{}.png".format(config.client_id,segment_type,
                                                         config.start_date,config.end_date,config.job_id))),dpi=1000)
    return


def binning(column, cut_points, labels=None):
    min_val = column.min()
    max_val = column.max()
    break_points = np.array([max_val] + cut_points + [min_val])
    column_bin = pd.cut(column, bins=np.sort(break_points), labels=labels[::-1], include_lowest=True)
    return column_bin


def save_dataframe_as_image(dataframe,path,config,segment_type):
    file_name = abspath(join(path,
              "client_id_{}_{}_summary_{}_{}_{}.png".format(config.client_id,segment_type,
                                                         config.start_date,config.end_date,config.job_id)))
    dfi.export(dataframe,filename = file_name,table_conversion = 'chrome')
    return


def generate_usage_segment(input_transaction_history, engine_handle):
    usage_agg_transaction_history = agg_rankingparam_by_customerlevelspecification(input_transaction_history,
                                                                                   global_config.usagesegment_customerlevel_specification)
    logging.info("Finished aggregating usage segment input_transaction_history by customer level specification")

    usage_segmentation_data = prep_segmentation_data(agg_transaction_history=usage_agg_transaction_history,
                                                     customer_level_specification=global_config.usagesegment_customerlevel_specification,
                                                     cohort_variable=global_config.usagesegment_rankingparam)

    features_to_create_segment = usage_segmentation_data.loc[usage_segmentation_data[global_config.usagesegment_rankingparam] > 0]
    features_to_create_segment = features_to_create_segment[['logRank', 'log' + global_config.usagesegment_rankingparam]]
    features_to_create_segment = features_to_create_segment.drop_duplicates(keep='first').reset_index(drop=True)
    # (x,y) co-ordinate of original curve
    usage_original_curve = np.column_stack((features_to_create_segment['logRank'],
                                            features_to_create_segment['log' + global_config.usagesegment_rankingparam]))

    min_slope_change = global_config.usagesegment_min_slope_change
    initial_epsilon_value = global_config.usagesegment_initial_epsilon_value
    required_segments = len(global_config.usagesegment_name)
    min_top_segment_rank = global_config.usagesegment_min_top_segment_rank
    epsilon_deviation = global_config.usagesegment_epsilon_deviation
    round_todigit = (-decimal.Decimal(str(epsilon_deviation)).as_tuple().exponent)

    created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(usage_original_curve, min_slope_change,
                                                                               initial_epsilon_value,
                                                                               min_top_segment_rank)
    logging.info(f"{created_segments} usage segments created at initial_epsilon_value")

    # If required segments less than the created segments at initial epsilon value, incrementing epsilon
    # until you get required number of segments
    if created_segments > required_segments:
        logging.info("Incrementing epsilon value to create required usage segments")
        incremented_epsilon_value = np.round((initial_epsilon_value + epsilon_deviation), round_todigit)
        while created_segments != required_segments:
            created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(usage_original_curve,
                                                                                       min_slope_change,
                                                                                       incremented_epsilon_value,
                                                                                       min_top_segment_rank)
            incremented_epsilon_value = np.round((incremented_epsilon_value + epsilon_deviation), round_todigit)
            if created_segments == 1:
                logging.info("Minimum segment value reached")
                break
    # If required segments greater than the created segments at initial epsilon value, decrementing epsilon
    # until you get required number of segments
    elif created_segments < required_segments:
        logging.info("Decrementing epsilon value to create required usage segments")
        decremented_epsilon_value = np.round((initial_epsilon_value - epsilon_deviation), round_todigit)
        while created_segments != required_segments:
            created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(usage_original_curve,
                                                                                       min_slope_change,
                                                                                       decremented_epsilon_value,
                                                                                       min_top_segment_rank)
            decremented_epsilon_value = np.round((decremented_epsilon_value - epsilon_deviation), round_todigit)
            if decremented_epsilon_value == 0:
                logging.info("Minimum epsilon value reached")
                break

    logging.info('Saving usage segment plot for inference')
    save_segment_plot(path=cache_plot_directory, config=global_config, original_curve=usage_original_curve,
                      sx=sx, sy=sy, idx=idx, cohort_variable=global_config.usagesegment_rankingparam,
                      segment_type="usageSegment")  # save to directory

    if created_segments != len(global_config.usagesegment_name):
        logging.info("required usage segments could not be created")
        return

    usage_segmentation_data['usageSegment'] = binning(column=usage_segmentation_data[global_config.usagesegment_rankingparam],
                                                      cut_points=y_cutoff_points,labels=list(global_config.usagesegment_name))
    usage_segmentation_data.rename(columns={'rankPercentile': 'usageSegmentValuePercentile'}, inplace=True)

    segment_summary = usage_segmentation_data.groupby('usageSegment').agg({global_config.usagesegment_rankingparam: ['min', 'max'],
                                                                           'usageSegmentValuePercentile': ['min', 'max'],
                                                                           'usageSegment': ['size']})
    segment_summary = segment_summary.sort_values([(global_config.usagesegment_rankingparam, 'min')], ascending=False)
    segment_summary['% customers'] = segment_summary['usageSegment']['size'] / len(usage_segmentation_data) * 100
    logging.info("Saving usage segment summary")
    save_dataframe_as_image(dataframe=segment_summary, path=cache_plot_directory, config=global_config, segment_type="usageSegment")

    usage_agg_transaction_history = usage_agg_transaction_history.merge(usage_segmentation_data[['purchasingEntityId',
                                                                                                 'usageSegmentValuePercentile',
                                                                                                 'usageSegment']],
                                                                        how='inner',on='purchasingEntityId')
    usage_agg_transaction_history['job_id'] = global_config.job_id
    usage_agg_transaction_history['usageSegmentRankingParam'] = global_config.usagesegment_rankingparam
    col_list = global_config.usagesegment_customerlevel_specification + [global_config.usagesegment_rankingparam] + ['usageSegment'] + ['usageSegmentValuePercentile']
    usage_segmentation_data = usage_segmentation_data[col_list]
    usage_segmentation_data = usage_segmentation_data.drop_duplicates(keep='first')
    # usage_segmentation_data.to_excel('usage_segmentation_data.xlsx',index = False)

    logging.info("Pushing segmentation output")
    save_segmentation_output(usage_agg_transaction_history, engine_handle=engine_handle)
    return


def generate_customer_segment(input_transaction_history, engine_handle):
    customer_agg_transaction_history = agg_rankingparam_by_customerlevelspecification(input_transaction_history,
                                                                                      global_config.customersegment_customerlevel_specification)
    logging.info("Finished aggregating customer segment input_transaction_history by customer level specification")

    customer_segmentation_data = prep_segmentation_data(agg_transaction_history=customer_agg_transaction_history,
                                                        customer_level_specification=global_config.customersegment_customerlevel_specification,
                                                        cohort_variable=global_config.customersegment_rankingparam)

    features_to_create_segment = customer_segmentation_data.loc[customer_segmentation_data[global_config.customersegment_rankingparam] > 0]
    features_to_create_segment = features_to_create_segment[['logRank', 'log' + global_config.customersegment_rankingparam]]
    features_to_create_segment = features_to_create_segment.drop_duplicates(keep='first').reset_index(drop=True)
    # (x,y) co-ordinate of original curve
    customer_original_curve = np.column_stack((features_to_create_segment['logRank'],
                                               features_to_create_segment['log' + global_config.customersegment_rankingparam]))

    min_slope_change = global_config.customersegment_min_slope_change
    initial_epsilon_value = global_config.customersegment_initial_epsilon_value
    required_segments = len(global_config.customersegment_name)
    min_top_segment_rank = global_config.customersegment_min_top_segment_rank
    epsilon_deviation = global_config.customersegment_epsilon_deviation
    round_todigit = (-decimal.Decimal(str(epsilon_deviation)).as_tuple().exponent)

    created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(customer_original_curve,
                                                                               min_slope_change, initial_epsilon_value,
                                                                               min_top_segment_rank)
    logging.info(f"{created_segments} customer segments created at initial_epsilon_value")
    # If required segments less than the created segments at initial epsilon value, incrementing epsilon
    # until you get required number of segments
    if created_segments > required_segments:
        logging.info("Incrementing epsilon value to create required customer segments")
        incremented_epsilon_value = np.round((initial_epsilon_value + epsilon_deviation), round_todigit)
        while created_segments != required_segments:
            created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(customer_original_curve,
                                                                                       min_slope_change,
                                                                                       incremented_epsilon_value,
                                                                                       min_top_segment_rank)
            incremented_epsilon_value = np.round((incremented_epsilon_value + epsilon_deviation), round_todigit)
            if created_segments == 1:
                logging.info("Minimum segment value reached")
                break
    # If required segments greater than the created segments at initial epsilon value, decrementing epsilon
    # until you get required number of segments
    elif created_segments < required_segments:
        logging.info("Decrementing epsilon value to create required customer segments")
        decremented_epsilon_value = np.round((initial_epsilon_value - epsilon_deviation), round_todigit)
        while created_segments != required_segments:
            created_segments, idx, y_cutoff_points, sx, sy = create_segment_at_epsilon(customer_original_curve,
                                                                                       min_slope_change,
                                                                                       decremented_epsilon_value,
                                                                                       min_top_segment_rank)
            decremented_epsilon_value = np.round((decremented_epsilon_value - epsilon_deviation), round_todigit)
            # Break the loop when epsilon value reaches 0
            if decremented_epsilon_value == 0:
                logging.info("Minimum epsilon value reached")
                break

    logging.info("Saving customer segment plot for inference")
    save_segment_plot(path=cache_plot_directory, config=global_config, original_curve=customer_original_curve,
                      sx=sx, sy=sy, idx=idx, cohort_variable=global_config.customersegment_rankingparam,
                      segment_type="customerSegment")  # save to directory

    if created_segments != len(global_config.customersegment_name):
        logging.info("required customer segments could not be created")
        return

    customer_segmentation_data['customerSegment'] = binning(column=customer_segmentation_data[global_config.customersegment_rankingparam],
                                                            cut_points=y_cutoff_points,labels=list(global_config.customersegment_name))
    customer_segmentation_data.rename(columns={'rankPercentile': 'customerSegmentValuePercentile'}, inplace=True)

    segment_summary = customer_segmentation_data.groupby('customerSegment').agg({global_config.customersegment_rankingparam: ['min', 'max'],
                                                                                 'customerSegmentValuePercentile': ['min', 'max'],
                                                                                 'customerSegment': ['size']})
    segment_summary = segment_summary.sort_values([(global_config.customersegment_rankingparam, 'min')],ascending=False)
    segment_summary['% customers'] = segment_summary['customerSegment']['size'] / len(customer_segmentation_data) * 100
    logging.info("Saving customer segment summary")
    save_dataframe_as_image(dataframe=segment_summary, path=cache_plot_directory, config=global_config, segment_type="customerSegment")

    customer_agg_transaction_history = customer_agg_transaction_history.merge(customer_segmentation_data[['purchasingEntityId',
                                                                                                          'customerSegmentValuePercentile',
                                                                                                          'customerSegment']],
                                                                              how='inner', on='purchasingEntityId')
    customer_agg_transaction_history['job_id'] = global_config.job_id
    customer_agg_transaction_history['customerSegmentRankingParam'] = global_config.customersegment_rankingparam
    col_list = global_config.customersegment_customerlevel_specification + [global_config.customersegment_rankingparam] + ['customerSegment'] + ['customerSegmentValuePercentile']
    customer_segmentation_data = customer_segmentation_data[col_list]
    customer_segmentation_data = customer_segmentation_data.drop_duplicates(keep='first')
    # customer_segmentation_data.to_excel('customer_segmentation_data.xlsx',index = False)

    logging.info("Pushing segmentation output")
    save_segmentation_output(customer_agg_transaction_history, engine_handle=engine_handle)
    return