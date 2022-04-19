import pandas as pd
import numpy as np
import scipy.stats 
from scipy.optimize import minimize


def get_ind_file(filetype):
    """
    Load and format the Ken French 30 Industry Portfolios files
    """
    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        sep = ','
        raise ValueError(f'filetype must be one of:{sep.join(known_types)}')
    if filetype == "returns":
        name = "vw_rets"
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    return get_ind_file("returns")

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms")

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size")

def drawdowns(return_series: pd.Series):
    """
    Takes a time series asset returns
    Computes and returns dataframe that contains:
    1. The wealth index
    2. The previous peaks
    3. Percentage drawdowns
    """
    wealth_index = (1+return_series).cumprod()
    prev_peaks = wealth_index.cummax()
    drawdown=(wealth_index-prev_peaks)/prev_peaks
    return(pd.DataFrame({
        'Wealth' : wealth_index,
        'Peaks' : prev_peaks,
        'Drawdown' : drawdown
    }))

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float of Series
    """
    demeaned_r = r-r.mean()
    #use population std, so set dof = 0
    sigma_r = r.std()
    exp = (demeaned_r**3).mean()
    return(exp/(sigma_r**3))

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float of Series
    """
    demeaned_r = r-r.mean()
    #use population std, so set dof = 0
    sigma_r = r.std()
    exp = (demeaned_r**4).mean()
    return(exp/(sigma_r**4))

def is_normal(r,level=0.01):
    """
    Applies Jarque bera test to determine whether a series is normal or not
    Test is applied at 1% signifinance level by default
    Returns True if hypothesis of normality if accepted, false otherwise
    """
    statistics, p_value = scipy.stats.jarque_bera(r)
    return(p_value > level)

def load_data(directory):
    """
    clean and load dataset into timeseries
    """
    df =pd.read_csv(directory, 
                     header=0, index_col = 0, parse_dates = True, na_values =-99.99)
    df = df/100
    df.index = pd.to_datetime(df.index, format = '%Y%m').to_period('M')
    return(df)

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std()

def var_historic(r, level = 5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return(r.aggregate(var_historic, level = level))
    elif isinstance(r, pd.Series):
        return(-np.percentile(r,level))
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
def cvar_historic(r, level = 5):
    """
    Computes the conditional VaR 
    """
    if isinstance(r, pd.DataFrame):
        is_beyond = r <= -var_historic(r, level=level)
        return(-r[is_beyond].mean())
    elif isinstance(r, pd.Series):
        return(r.aggregate(cvar_historic, level=level))
    else:
        raise TypeError("Expected r to be Series or DataFrame")

        
def var_gaussian(r, level = 5, modified = False):
    """
    Returns Parametric Gaussian VaR of a Series DataFrame
    """
    #compute the z score assuming it was gaussian
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and Kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 -1)*s/6 +
                 (z**3 -3*z) * (k-3)/24 -
                 (2*z**3 - 5*z) *(s**2)/36
            )
        
    return(-(r.mean()+z*r.std()))

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def ef_table(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Table of weights, return and volatility of efficient portfolio at different points.
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ef[list(er.index)] = weights
    return(ef)

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

