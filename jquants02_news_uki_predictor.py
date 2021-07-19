# -*- coding: utf-8 -*-
import io
import os
import pickle
import datetime

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from tqdm.auto import tqdm

class ScoringService(object):
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    dfs = None
    models = None
    codes = None
    
    @classmethod
    def get_inputs(cls, dataset_dir):
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs
    
    """@classmethod
    def get_dataset(cls, inputs):
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            #if k not in ["stock_fin_price", "stock_labels"]:
            if k not in ["stock_fin_price"]:
                cls.dfs[k] = pd.read_csv(v)
        return cls.dfs"""
        
    @classmethod
    def get_dataset(cls, inputs, load_data):
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            if k not in load_data:
                continue
            cls.dfs[k] = pd.read_csv(v)
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(cls.dfs[k].loc[:, "EndOfDayQuote Date"])
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(cls.dfs[k].loc[:, "base_date"])
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs
    
    """@classmethod
    def get_codes(cls, dfs):
        stock_list = dfs["stock_list"].copy()
        cls.codes = stock_list[stock_list["prediction_target"] == True]["Local Code"].values
        return cls.codes"""
        
    @classmethod
    def get_codes(cls, dfs):
        stock_list = dfs["stock_list"].copy()
        cls.codes = stock_list[stock_list["universe_comp2"] == True]["Local Code"].values
        return cls.codes
    
    @classmethod
    def get_technical(cls, dfs, code):
        tmp_df = dfs["stock_price"][dfs["stock_price"]["Local Code"]==code].copy()
        
        # 終値
        tmp_df["close"] = tmp_df["EndOfDayQuote Close"]
        
        # 騰落率
        tmp_df["ror_1"] = tmp_df["EndOfDayQuote Close"].pct_change(1)
        tmp_df["ror_5"] = tmp_df["EndOfDayQuote Close"].pct_change(5)
        tmp_df["ror_10"] = tmp_df["EndOfDayQuote Close"].pct_change(10)
        tmp_df["ror_20"] = tmp_df["EndOfDayQuote Close"].pct_change(20)
        tmp_df["ror_40"] = tmp_df["EndOfDayQuote Close"].pct_change(40)
        tmp_df["ror_60"] = tmp_df["EndOfDayQuote Close"].pct_change(60)
        tmp_df["ror_100"] = tmp_df["EndOfDayQuote Close"].pct_change(100)
        
        # 売買代金
        tmp_df["volume"] = tmp_df["EndOfDayQuote Close"] * tmp_df["EndOfDayQuote Volume"]
        tmp_df = tmp_df.replace([np.inf, -np.inf], np.nan)
        
        tmp_df["vol_1"] = tmp_df["volume"]
        tmp_df["vol_5"] = tmp_df["volume"].rolling(5).mean()
        tmp_df["vol_10"] = tmp_df["volume"].rolling(10).mean()
        tmp_df["vol_20"] = tmp_df["volume"].rolling(20).mean()
        tmp_df["vol_40"] = tmp_df["volume"].rolling(40).mean()
        tmp_df["vol_60"] = tmp_df["volume"].rolling(60).mean()
        tmp_df["vol_100"] = tmp_df["volume"].rolling(100).mean()
        tmp_df["d_vol"] = tmp_df["volume"]/tmp_df["vol_20"]
        
        # レンジ
        tmp_df["range"] = (tmp_df[["EndOfDayQuote PreviousClose", "EndOfDayQuote High"]].max(axis=1) - tmp_df[["EndOfDayQuote PreviousClose", "EndOfDayQuote Low"]].min(axis=1)) / tmp_df["EndOfDayQuote PreviousClose"]
        tmp_df = tmp_df.replace([np.inf, -np.inf], np.nan)
        
        tmp_df["atr_1"] = tmp_df["range"]
        tmp_df["atr_5"] = tmp_df["range"].rolling(5).mean()
        tmp_df["atr_10"] = tmp_df["range"].rolling(10).mean()
        tmp_df["atr_20"] = tmp_df["range"].rolling(20).mean()
        tmp_df["atr_40"] = tmp_df["range"].rolling(40).mean()
        tmp_df["atr_60"] = tmp_df["range"].rolling(60).mean()
        tmp_df["atr_100"] = tmp_df["range"].rolling(100).mean()
        tmp_df["d_atr"] = tmp_df["range"]/tmp_df["atr_20"]
        
        # ギャップレンジ
        tmp_df["gap_range"] = (np.abs(tmp_df["EndOfDayQuote Open"] - tmp_df["EndOfDayQuote PreviousClose"])) / tmp_df["EndOfDayQuote PreviousClose"]
        tmp_df["g_atr_1"] = tmp_df["gap_range"]
        tmp_df["g_atr_5"] = tmp_df["gap_range"].rolling(5).mean()
        tmp_df["g_atr_10"] = tmp_df["gap_range"].rolling(10).mean()
        tmp_df["g_atr_20"] = tmp_df["gap_range"].rolling(20).mean()
        tmp_df["g_atr_40"] = tmp_df["gap_range"].rolling(40).mean()
        tmp_df["g_atr_60"] = tmp_df["gap_range"].rolling(60).mean()
        tmp_df["g_atr_100"] = tmp_df["gap_range"].rolling(100).mean()
        
        # デイレンジ
        tmp_df["day_range"] = (tmp_df["EndOfDayQuote High"] - tmp_df["EndOfDayQuote Low"]) / tmp_df["EndOfDayQuote PreviousClose"]
        tmp_df["d_atr_1"] = tmp_df["day_range"]
        tmp_df["d_atr_5"] = tmp_df["day_range"].rolling(5).mean()
        tmp_df["d_atr_10"] = tmp_df["day_range"].rolling(10).mean()
        tmp_df["d_atr_20"] = tmp_df["day_range"].rolling(20).mean()
        tmp_df["d_atr_40"] = tmp_df["day_range"].rolling(40).mean()
        tmp_df["d_atr_60"] = tmp_df["day_range"].rolling(60).mean()
        tmp_df["d_atr_100"] = tmp_df["day_range"].rolling(100).mean()
        
        # ヒゲレンジ
        tmp_df["hig_range"] = ((tmp_df["EndOfDayQuote High"] - tmp_df["EndOfDayQuote Low"]) - np.abs(tmp_df["EndOfDayQuote Open"] - tmp_df["EndOfDayQuote Close"])) / tmp_df["EndOfDayQuote PreviousClose"]
        tmp_df["h_atr_1"] = tmp_df["hig_range"]
        tmp_df["h_atr_5"] = tmp_df["hig_range"].rolling(5).mean()
        tmp_df["h_atr_10"] = tmp_df["hig_range"].rolling(10).mean()
        tmp_df["h_atr_20"] = tmp_df["hig_range"].rolling(20).mean()
        tmp_df["h_atr_40"] = tmp_df["hig_range"].rolling(40).mean()
        tmp_df["h_atr_60"] = tmp_df["hig_range"].rolling(60).mean()
        tmp_df["h_atr_100"] = tmp_df["hig_range"].rolling(100).mean()
        
        # ボラティリティ
        tmp_df["vola_5"] = tmp_df["ror_1"].rolling(5).std()
        tmp_df["vola_10"] = tmp_df["ror_1"].rolling(10).std()
        tmp_df["vola_20"] = tmp_df["ror_1"].rolling(20).std()
        tmp_df["vola_40"] = tmp_df["ror_1"].rolling(40).std()
        tmp_df["vola_60"] = tmp_df["ror_1"].rolling(60).std()
        tmp_df["vola_100"] = tmp_df["ror_1"].rolling(100).std()
        
        # HLバンド
        tmp_df["hl_5"] = tmp_df["EndOfDayQuote High"].rolling(5).max() - tmp_df["EndOfDayQuote Low"].rolling(5).min()
        tmp_df["hl_10"] = tmp_df["EndOfDayQuote High"].rolling(10).max() - tmp_df["EndOfDayQuote Low"].rolling(10).min()
        tmp_df["hl_20"] = tmp_df["EndOfDayQuote High"].rolling(20).max() - tmp_df["EndOfDayQuote Low"].rolling(20).min()
        tmp_df["hl_40"] = tmp_df["EndOfDayQuote High"].rolling(40).max() - tmp_df["EndOfDayQuote Low"].rolling(40).min()
        tmp_df["hl_60"] = tmp_df["EndOfDayQuote High"].rolling(60).max() - tmp_df["EndOfDayQuote Low"].rolling(60).min()
        tmp_df["hl_100"] = tmp_df["EndOfDayQuote High"].rolling(100).max() - tmp_df["EndOfDayQuote Low"].rolling(100).min()
        
        # マーケットインパクト
        tmp_df["mi"] = tmp_df["range"] / (tmp_df["EndOfDayQuote Volume"] * tmp_df["EndOfDayQuote Close"])
        tmp_df = tmp_df.replace([np.inf, -np.inf], np.nan)
        
        tmp_df["mi_5"] = tmp_df["mi"].rolling(5).mean()
        tmp_df["mi_10"] = tmp_df["mi"].rolling(10).mean()
        tmp_df["mi_20"] = tmp_df["mi"].rolling(20).mean()
        tmp_df["mi_40"] = tmp_df["mi"].rolling(40).mean()
        tmp_df["mi_60"] = tmp_df["mi"].rolling(60).mean()
        tmp_df["mi_100"] = tmp_df["mi"].rolling(100).mean()
        
        feat = ["EndOfDayQuote Date", "Local Code", "close", 
                "ror_1", "ror_5", "ror_10", "ror_20", "ror_40", "ror_60", "ror_100",
                "vol_1", "vol_5", "vol_10", "vol_20", "vol_40", "vol_60", "vol_100", "d_vol",
                "atr_1", "atr_5", "atr_10", "atr_20", "atr_40", "atr_60", "atr_100", "d_atr",
                "g_atr_1", "g_atr_5", "g_atr_10", "g_atr_20", "g_atr_40", "g_atr_60", "g_atr_100",
                "d_atr_1", "d_atr_5", "d_atr_10", "d_atr_20", "d_atr_40", "d_atr_60", "d_atr_100",
                "h_atr_1", "h_atr_5", "h_atr_10", "h_atr_20", "h_atr_40", "h_atr_60", "h_atr_100",
                "vola_5", "vola_10", "vola_20", "vola_40", "vola_60", "vola_100",
                "hl_5", "hl_10", "hl_20", "hl_40", "hl_60", "hl_100",
                "mi_5", "mi_10", "mi_20", "mi_40", "mi_60", "mi_100"]
        tmp_df = tmp_df[feat]
        tmp_df.columns = ["datetime", "code", "close", 
                          "ror_1", "ror_5", "ror_10", "ror_20", "ror_40", "ror_60", "ror_100",
                          "vol_1", "vol_5", "vol_10", "vol_20", "vol_40", "vol_60", "vol_100", "d_vol",
                          "atr_1", "atr_5", "atr_10", "atr_20", "atr_40", "atr_60", "atr_100", "d_atr",
                          "g_atr_1", "g_atr_5", "g_atr_10", "g_atr_20", "g_atr_40", "g_atr_60", "g_atr_100",
                          "d_atr_1", "d_atr_5", "d_atr_10", "d_atr_20", "d_atr_40", "d_atr_60", "d_atr_100",
                          "h_atr_1", "h_atr_5", "h_atr_10", "h_atr_20", "h_atr_40", "h_atr_60", "h_atr_100",
                          "vola_5", "vola_10", "vola_20", "vola_40", "vola_60", "vola_100",
                          "hl_5", "hl_10", "hl_20", "hl_40", "hl_60", "hl_100",
                          "mi_5", "mi_10", "mi_20", "mi_40", "mi_60", "mi_100"]
        tmp_df["datetime"] = pd.to_datetime(tmp_df["datetime"])
        tmp_df = tmp_df.set_index(["datetime", "code"])
        return tmp_df
        
    @classmethod
    def get_financial(cls, dfs, code):
        tmp_df = dfs["stock_fin"][dfs["stock_fin"]["Local Code"]==code].copy()
        tmp_df = tmp_df.ffill()
        
        # 本決算／中間決算フラグ、修正開示フラグ、事後修正有無フラグ
        tmp_df["annual"] = 0
        tmp_df["revision"] = 0
        tmp_df.loc[tmp_df["Result_FinancialStatement ReportType"]=="Annual", "annual"] = 1
        tmp_df.loc[tmp_df["Result_FinancialStatement ModifyDate"]==tmp_df["Result_FinancialStatement ModifyDate"].shift(1), "revision"] = 1
        
        feat1 = ["annual", "revision"]
        
        # 原系列
        tmp_df["pre_result_period_end"] = tmp_df["Result_FinancialStatement FiscalPeriodEnd"].shift(1)
        tmp_df["r_sales"] = np.nan
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] != "Q1") ), "r_sales"] = tmp_df["Result_FinancialStatement NetSales"].diff(1)
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] == "Q1") ), "r_sales"] = tmp_df["Result_FinancialStatement NetSales"]
        tmp_df["r_sales"] = tmp_df["r_sales"].ffill()
        tmp_df["r_ope_income"] = np.nan
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] != "Q1") ), "r_ope_income"] = tmp_df["Result_FinancialStatement OperatingIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] == "Q1") ), "r_ope_income"] = tmp_df["Result_FinancialStatement OperatingIncome"]
        tmp_df["r_ope_income"] = tmp_df["r_ope_income"].ffill()
        tmp_df["r_ord_income"] = np.nan
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] != "Q1") ), "r_ord_income"] = tmp_df["Result_FinancialStatement OrdinaryIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] == "Q1") ), "r_ord_income"] = tmp_df["Result_FinancialStatement OrdinaryIncome"]
        tmp_df["r_ord_income"] = tmp_df["r_ord_income"].ffill()
        tmp_df["r_net_income"] = np.nan
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] != "Q1") ), "r_net_income"] = tmp_df["Result_FinancialStatement NetIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Result_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_result_period_end"]) & (tmp_df["Result_FinancialStatement ReportType"] == "Q1") ), "r_net_income"] = tmp_df["Result_FinancialStatement NetIncome"]
        tmp_df["r_net_income"] = tmp_df["r_net_income"].ffill()
        
        tmp_df["pre_forecast_period_end"] = tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"].shift(1)
        tmp_df["f_sales"] = np.nan
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] != "Q1") ), "f_sales"] = tmp_df["Forecast_FinancialStatement NetSales"].diff(1)
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] == "Q1") ), "f_sales"] = tmp_df["Forecast_FinancialStatement NetSales"]
        tmp_df["f_sales"] = tmp_df["f_sales"].ffill()
        tmp_df["f_ope_income"] = np.nan
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] != "Q1") ), "f_ope_income"] = tmp_df["Forecast_FinancialStatement OperatingIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] == "Q1") ), "f_ope_income"] = tmp_df["Forecast_FinancialStatement OperatingIncome"]
        tmp_df["f_ope_income"] = tmp_df["f_ope_income"].ffill()
        tmp_df["f_ord_income"] = np.nan
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] != "Q1") ), "f_ord_income"] = tmp_df["Forecast_FinancialStatement OrdinaryIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] == "Q1") ), "f_ord_income"] = tmp_df["Forecast_FinancialStatement OrdinaryIncome"]
        tmp_df["f_ord_income"] = tmp_df["f_ord_income"].ffill()
        tmp_df["f_net_income"] = np.nan
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] != "Q1") ), "f_net_income"] = tmp_df["Forecast_FinancialStatement NetIncome"].diff(1)
        tmp_df.loc[( (tmp_df["Forecast_FinancialStatement FiscalPeriodEnd"] != tmp_df["pre_forecast_period_end"]) & (tmp_df["Forecast_FinancialStatement ReportType"] == "Q1") ), "f_net_income"] = tmp_df["Forecast_FinancialStatement NetIncome"]
        tmp_df["f_net_income"] = tmp_df["f_net_income"].ffill()
        
        tmp_df["r_expense1"] = tmp_df["r_sales"] - tmp_df["r_ope_income"]
        tmp_df["r_expense2"] = tmp_df["r_ope_income"] - tmp_df["r_ord_income"]
        tmp_df["r_expense3"] = tmp_df["r_ord_income"] - tmp_df["r_net_income"]
        
        tmp_df["f_expense1"] = tmp_df["f_sales"] - tmp_df["f_ope_income"]
        tmp_df["f_expense2"] = tmp_df["f_ope_income"] - tmp_df["f_ord_income"]
        tmp_df["f_expense3"] = tmp_df["f_ord_income"] - tmp_df["f_net_income"]
        
        tmp_df["r_assets"] = tmp_df["Result_FinancialStatement TotalAssets"]
        tmp_df["r_equity"] = tmp_df["Result_FinancialStatement NetAssets"]
        
        tmp_df["operating_cf"] = tmp_df["Result_FinancialStatement CashFlowsFromOperatingActivities"]
        tmp_df["financial_cf"] = tmp_df["Result_FinancialStatement CashFlowsFromFinancingActivities"]
        tmp_df["investing_cf"] = tmp_df["Result_FinancialStatement CashFlowsFromInvestingActivities"]
        
        feat2 = ["r_sales", "r_ope_income", "r_ord_income", "r_net_income", "f_sales", "f_ope_income", "f_ord_income", "f_net_income",
                 "r_expense1", "r_expense2", "r_expense3", "f_expense1", "f_expense2", "f_expense3",
                 "r_assets", "r_equity", "operating_cf", "financial_cf", "investing_cf"]
        
        # 複合指標　原系列
        # 純利益系
        tmp_df["r_pm1"]  = tmp_df["Result_FinancialStatement NetIncome"]   / tmp_df["Result_FinancialStatement NetSales"]
        tmp_df["r_roe1"] = tmp_df["Result_FinancialStatement NetIncome"]   / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["r_roa1"] = tmp_df["Result_FinancialStatement NetIncome"]   / tmp_df["Result_FinancialStatement TotalAssets"]
        
        tmp_df["f_pm1"]  = tmp_df["Forecast_FinancialStatement NetIncome"] / tmp_df["Forecast_FinancialStatement NetSales"]
        tmp_df["f_roe1"] = tmp_df["Forecast_FinancialStatement NetIncome"] / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["f_roa1"] = tmp_df["Forecast_FinancialStatement NetIncome"] / tmp_df["Result_FinancialStatement TotalAssets"]
        
        # 経常利益系
        tmp_df["r_pm2"]  = tmp_df["Result_FinancialStatement OrdinaryIncome"]   / tmp_df["Result_FinancialStatement NetSales"]
        tmp_df["r_roe2"] = tmp_df["Result_FinancialStatement OrdinaryIncome"]   / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["r_roa2"] = tmp_df["Result_FinancialStatement OrdinaryIncome"]   / tmp_df["Result_FinancialStatement TotalAssets"]
        
        tmp_df["f_pm2"]  = tmp_df["Forecast_FinancialStatement OrdinaryIncome"] / tmp_df["Forecast_FinancialStatement NetSales"]
        tmp_df["f_roe2"] = tmp_df["Forecast_FinancialStatement OrdinaryIncome"] / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["f_roa2"] = tmp_df["Forecast_FinancialStatement OrdinaryIncome"] / tmp_df["Result_FinancialStatement TotalAssets"]
        
        # 営業利益系
        tmp_df["r_pm3"]  = tmp_df["Result_FinancialStatement OperatingIncome"]   / tmp_df["Result_FinancialStatement NetSales"]
        tmp_df["r_roe3"] = tmp_df["Result_FinancialStatement OperatingIncome"]   / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["r_roa3"] = tmp_df["Result_FinancialStatement OperatingIncome"]   / tmp_df["Result_FinancialStatement TotalAssets"]
        
        tmp_df["f_pm3"]  = tmp_df["Forecast_FinancialStatement OperatingIncome"] / tmp_df["Forecast_FinancialStatement NetSales"]
        tmp_df["f_roe3"] = tmp_df["Forecast_FinancialStatement OperatingIncome"] / tmp_df["Result_FinancialStatement NetAssets"]
        tmp_df["f_roa3"] = tmp_df["Forecast_FinancialStatement OperatingIncome"] / tmp_df["Result_FinancialStatement TotalAssets"]
        
        # コスト
        tmp_df["r_cost1"] = ((tmp_df["Result_FinancialStatement NetSales"] - tmp_df["Result_FinancialStatement OperatingIncome"])/tmp_df["Result_FinancialStatement NetSales"])
        tmp_df["r_cost2"] = ((tmp_df["Result_FinancialStatement OperatingIncome"] - tmp_df["Result_FinancialStatement OrdinaryIncome"])/tmp_df["Result_FinancialStatement NetSales"])
        tmp_df["r_cost3"] = ((tmp_df["Result_FinancialStatement OrdinaryIncome"] - tmp_df["Result_FinancialStatement NetIncome"])/tmp_df["Result_FinancialStatement NetSales"])
        
        tmp_df["f_cost1"] = ((tmp_df["Forecast_FinancialStatement NetSales"] - tmp_df["Forecast_FinancialStatement OperatingIncome"])/tmp_df["Forecast_FinancialStatement NetSales"])
        tmp_df["f_cost2"] = ((tmp_df["Forecast_FinancialStatement OperatingIncome"] - tmp_df["Forecast_FinancialStatement OrdinaryIncome"])/tmp_df["Forecast_FinancialStatement NetSales"])
        tmp_df["f_cost3"] = ((tmp_df["Forecast_FinancialStatement OrdinaryIncome"] - tmp_df["Forecast_FinancialStatement NetIncome"])/tmp_df["Forecast_FinancialStatement NetSales"])
        
        # 売上高回転率
        tmp_df["r_turn"] = tmp_df["Result_FinancialStatement NetSales"] / tmp_df["Result_FinancialStatement TotalAssets"]
        tmp_df["f_turn"] = tmp_df["Forecast_FinancialStatement NetSales"] / tmp_df["Result_FinancialStatement TotalAssets"]
        
        # 財務健全性
        tmp_df["equity_ratio"] = tmp_df["Result_FinancialStatement NetAssets"] / tmp_df["Result_FinancialStatement TotalAssets"]
        
        # 総資本キャッシュフロー比率
        tmp_df["o_cf_ratio"] = (tmp_df["Result_FinancialStatement CashFlowsFromOperatingActivities"]/tmp_df["Result_FinancialStatement TotalAssets"])
        tmp_df["f_cf_ratio"] = (tmp_df["Result_FinancialStatement CashFlowsFromFinancingActivities"]/tmp_df["Result_FinancialStatement TotalAssets"])
        tmp_df["i_cf_ratio"] = (tmp_df["Result_FinancialStatement CashFlowsFromInvestingActivities"]/tmp_df["Result_FinancialStatement TotalAssets"])
        
        feat3 = ["r_pm1", "r_roe1", "r_roa1", "f_pm1", "f_roe1", "f_roa1", 
                 "r_pm2", "r_roe2", "r_roa2", "f_pm2", "f_roe2", "f_roa2",
                 "r_pm3", "r_roe3", "r_roa3", "f_pm3", "f_roe3", "f_roa3",
                 "r_cost1", "r_cost2", "r_cost3", "f_cost1", "f_cost2", "f_cost3",
                 "r_turn", "f_turn", "equity_ratio", "o_cf_ratio", "f_cf_ratio", "i_cf_ratio"
                ]
        
        # Inf値をNan値化
        tmp_df = tmp_df.replace([np.inf, -np.inf], np.nan)
        
        # 差分系列
        d_feat2 = []
        
        for f in feat2:
            tmp_df["d_"+f] = tmp_df[f].diff(1)
            d_feat2.append("d_"+f)
        
        d_feat3 = []
        for f in feat3:
            tmp_df["d_"+f] = tmp_df[f].diff(1)
            d_feat3.append("d_"+f)
        
        d_feat4 = ["m_sales", "m_ope_income", "m_ord_income", "m_net_income", "m_expense1", "m_expense2", "m_expense3",
                   "m_pm1", "m_pm2", "m_pm3", "m_roe1", "m_roe2", "m_roe3", "m_roa1", "m_roa2", "m_roa3",
                   "m_cost1", "m_cost2", "m_cost3"]
        
        tmp_df["m_sales"] = tmp_df["r_sales"] - tmp_df["f_sales"].shift(1)
        tmp_df["m_ope_income"] = tmp_df["r_ope_income"] - tmp_df["f_ope_income"].shift(1)
        tmp_df["m_ord_income"] = tmp_df["r_ord_income"] - tmp_df["f_ord_income"].shift(1)
        tmp_df["m_net_income"] = tmp_df["r_net_income"] - tmp_df["f_net_income"].shift(1)
        tmp_df["m_expense1"] = tmp_df["r_expense1"] - tmp_df["f_expense1"].shift(1)
        tmp_df["m_expense2"] = tmp_df["r_expense2"] - tmp_df["f_expense2"].shift(1)
        tmp_df["m_expense3"] = tmp_df["r_expense3"] - tmp_df["f_expense3"].shift(1)
        
        tmp_df["m_pm1"] = tmp_df["r_pm1"] - tmp_df["f_pm1"].shift(1)
        tmp_df["m_pm2"] = tmp_df["r_pm2"] - tmp_df["f_pm2"].shift(1)
        tmp_df["m_pm3"] = tmp_df["r_pm3"] - tmp_df["f_pm3"].shift(1)
        tmp_df["m_roe1"] = tmp_df["r_roe1"] - tmp_df["f_roe1"].shift(1)
        tmp_df["m_roe2"] = tmp_df["r_roe2"] - tmp_df["f_roe2"].shift(1)
        tmp_df["m_roe3"] = tmp_df["r_roe3"] - tmp_df["f_roe3"].shift(1)
        tmp_df["m_roa1"] = tmp_df["r_roa1"] - tmp_df["f_roa1"].shift(1)
        tmp_df["m_roa2"] = tmp_df["r_roa2"] - tmp_df["f_roa2"].shift(1)
        tmp_df["m_roa3"] = tmp_df["r_roa3"] - tmp_df["f_roa3"].shift(1)
        tmp_df["m_cost1"] = tmp_df["r_cost1"] - tmp_df["f_cost1"].shift(1)
        tmp_df["m_cost2"] = tmp_df["r_cost2"] - tmp_df["f_cost2"].shift(1)
        tmp_df["m_cost3"] = tmp_df["r_cost3"] - tmp_df["f_cost3"].shift(1)    
        
        feat = ["base_date", "Local Code"]
        feat.extend(feat1)
        feat.extend(feat2)
        feat.extend(feat3)
        feat.extend(d_feat2)
        feat.extend(d_feat3)
        feat.extend(d_feat4)
        
        col_names = ["datetime", "code"]
        col_names.extend(feat1)
        col_names.extend(feat2)
        col_names.extend(feat3)
        col_names.extend(d_feat2)
        col_names.extend(d_feat3)
        col_names.extend(d_feat4)
        
        tmp_df = tmp_df[feat]
        tmp_df.columns = col_names
        tmp_df["datetime"] = pd.to_datetime(tmp_df["datetime"])
        tmp_df = tmp_df.set_index(["datetime", "code"])
        return tmp_df
        
    """@classmethod
    def get_df_merge(cls, dfs, train=True):
        df_technical = []
        for code in dfs["stock_fin"]["Local Code"].unique():
            df_technical.append(cls.get_technical(dfs, code))
        df_technical = pd.concat(df_technical)
        
        df_financial = []
        for code in dfs["stock_fin"]["Local Code"].unique():
            df_financial.append(cls.get_financial(dfs, code))
        df_financial = pd.concat(df_financial)
        
        if train:
            df_label = dfs["stock_labels"].copy()
            feat = ["base_date", "Local Code", "label_high_20", "label_low_20"]
            df_label = df_label[feat]
            df_label.columns = ["datetime", "code", "label_high_20", "label_low_20"]
            df_label["datetime"] = pd.to_datetime(df_label["datetime"])
            df_label = df_label.set_index(["datetime", "code"])
            
            df_merge = pd.concat([df_financial,
                                  df_technical[df_technical.index.isin(df_financial.index)],
                                  df_label[df_label.index.isin(df_financial.index)]
                                 ], axis=1)
        else:
            df_merge = pd.concat([df_financial,
                                  df_technical[df_technical.index.isin(df_financial.index)],
                                 ], axis=1)
        
        df_merge = df_merge.reset_index()
        return df_merge"""
        
    @classmethod
    def get_df_merge(cls, dfs, train=True):
        df_technical = []
        for code in cls.codes:
            df_technical.append(cls.get_technical(dfs, code))
        df_technical = pd.concat(df_technical)
        
        df_financial = []
        for code in cls.codes:
            df_financial.append(cls.get_financial(dfs, code))
        df_financial = pd.concat(df_financial)
        
        if train:
            df_label = dfs["stock_labels"].copy()
            feat = ["base_date", "Local Code", "label_high_20", "label_low_20"]
            df_label = df_label[feat]
            df_label.columns = ["datetime", "code", "label_high_20", "label_low_20"]
            df_label["datetime"] = pd.to_datetime(df_label["datetime"])
            df_label = df_label.set_index(["datetime", "code"])
            
            df_merge = pd.concat([df_financial,
                                  df_technical[df_technical.index.isin(df_financial.index)],
                                  df_label[df_label.index.isin(df_financial.index)]
                                 ], axis=1)
        else:
            df_merge = pd.concat([df_financial,
                                  df_technical[df_technical.index.isin(df_financial.index)],
                                 ], axis=1)
        
        df_merge = df_merge.reset_index()
        return df_merge
        
    @classmethod
    def get_df_for_ml(cls, dfs, train=True):
        df_merge = cls.get_df_merge(dfs, train=train)
        df_merge = df_merge.replace([np.inf, -np.inf], np.nan)
        df_merge = df_merge.fillna(0)
        return df_merge
        
    @classmethod
    def get_recent_statements(cls, df, dt):
        cnt = 1
        while True:
            tmp_df = df[(df["datetime"] > dt-datetime.timedelta(cnt)) & (df["datetime"] <= dt)].copy()
            if len(tmp_df) >= 100:
                break
            cnt += 1
        tmp_df["datetime"] = dt
        return tmp_df
    
    @classmethod
    def get_df_for_predict(cls, dfs, train=True):
        df_merge = cls.get_df_merge(dfs, train=train)
        df_merge = df_merge.replace([np.inf, -np.inf], np.nan)
        df_merge = df_merge.fillna(0)
        dt_list = [pd.Timestamp("2020-01-01") + datetime.timedelta(i) for i in range((dfs["stock_fin"].index.max()-pd.Timestamp("2020-01-01")).days + 1)]
        dt_list = pd.Series(dt_list)
        FRIDAY = 4
        dt_list = dt_list[dt_list.dt.dayofweek == FRIDAY]
        tmp_dfs = []
        for dts in dt_list:
            tmp = cls.get_recent_statements(df_merge, dts)
            tmp_dfs.append(tmp)
        df_merge = pd.concat(tmp_dfs)
        df_merge = df_merge[df_merge["datetime"]>="2020-01-01"]
        return df_merge
        
    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = ["model_h_final", "model_l_final"]
        for label in labels:
            m = os.path.join(model_path, f"my_model_{label}.pkl")
            with open(m, "rb") as f:
                cls.models[label] = pickle.load(f)
        return True
        
    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            pickle.dump(model, f)
        
    @classmethod
    def get_predict(cls, df_for_ml):
        tmp_df = df_for_ml.copy()
        
        x_feats = [f for f in tmp_df.columns if f not in ["datetime", "code", "label_high_20", "label_low_20"]]
        
        tmp_df["pred_high"] = cls.models["model_h_final"].predict(tmp_df[x_feats])
        tmp_df["pred_low"]  = cls.models["model_l_final"].predict(tmp_df[x_feats])
        
        tmp_df = tmp_df.set_index("datetime")
        cols = ["code", "pred_high", "pred_low"]
        tmp_df = tmp_df[cols]
        tmp_df.columns = ["code", "label_high_20", "label_low_20"]
        
        return tmp_df
        
    @classmethod
    def train_and_save_model(cls, inputs, labels=None, codes=None, model_path="../model"):
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        
        # 価格データの修正
        cls.dfs["stock_price"]["EndOfDayQuote Open"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote High"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Low"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Close"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        
        # 特徴量を作成
        df_for_ml = cls.get_df_for_ml(cls.dfs, train=True)
        
        TRAIN_START = "2017-01-01"
        TRAIN_END   = "2019-12-31"
        TEST_START  = "2020-01-01"
        TEST_END    = "2020-11-15"
        
        train_df = df_for_ml[df_for_ml["datetime"] <= TRAIN_END].copy()
        
        model_h_final = XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=3000, n_jobs=-1, colsample_bytree=0.1, random_state=0)
        model_l_final = XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=3000, n_jobs=-1, colsample_bytree=0.1, random_state=0)
        
        x_feats = [f for f in df_for_ml.columns if f not in ["datetime", "code", "label_high_20", "label_low_20"]]
        y_labels = ["label_high_20", "label_low_20"]
        
        model_h_final.fit(train_df[x_feats], train_df["label_high_20"])
        model_l_final.fit(train_df[x_feats], train_df["label_low_20"])
        
        cls.save_model(model_h_final, "model_h_final", model_path=model_path)
        cls.save_model(model_l_final, "model_l_final", model_path=model_path)
        
    """@classmethod
    def predict(cls, inputs, labels=None, codes=None):
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        
        # 価格データの修正
        cls.dfs["stock_price"]["EndOfDayQuote Open"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote High"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Low"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Close"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        
        # 特徴量を作成
        df_for_ml = cls.get_df_for_ml(cls.dfs, train=False)
        
        # 訓練および予測
        df = cls.get_predict(df_for_ml)
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)
        
        # 出力対象列を定義
        output_columns = ["code", "label_high_20", "label_low_20"]
        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)
        #df.to_csv("test_submit.csv", index=False)
        
        return out.getvalue()"""
        
    @classmethod
    def predict(cls, inputs, labels=None, codes=None, load_data=["stock_list", "stock_fin", "stock_price"]):
        if cls.dfs is None:
            cls.get_dataset(inputs, ["stock_list", "stock_fin", "stock_price"])
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        
        # 価格データの修正
        cls.dfs["stock_price"]["EndOfDayQuote Open"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote High"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Low"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        cls.dfs["stock_price"]["EndOfDayQuote Close"][cls.dfs["stock_price"]["EndOfDayQuote Close"]==0] = cls.dfs["stock_price"]["EndOfDayQuote ExchangeOfficialClose"]
        
        # 特徴量を作成
        df_for_ml = cls.get_df_for_predict(cls.dfs, train=False)
        
        # 訓練および予測
        df = cls.get_predict(df_for_ml)
        
        df.index.name = "date"
        df.reset_index(inplace=True)
        df.rename(columns={"label_high_20":"pred", "code":"Local Code"}, inplace=True)
        df = df.sort_values("pred", ascending=False)
        
        df = df.groupby("date").head(10)
        df.set_index("date", inplace=True)
        df.sort_index(kind="mergesort", inplace=True)
        df.index = df.index + pd.Timedelta("3D")
        
        df.reset_index(inplace=True)
        df.loc[:, "budget"] = 200000
        df["Local Code"] = df["Local Code"].astype("int")
        output_columns = ["date", "Local Code", "budget"]
        
        out = io.StringIO()
        df.to_csv(out, header=True, index=False, columns=output_columns)
        #df.to_csv("test_submit.csv", index=False)
        
        return out.getvalue()