from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import pprint as pp
from utils.feature_builder import load_data, create_dataframe
import os

print(os.cpu_count())

data = load_data('data/data.pkl')
df = create_dataframe(data)


def test_model(df):
    # Lag the "flow" variable by one day
    df['flow_lagged'] = df['flow'].shift(1)

    # Remove the first row, which will have a NaN value for the lagged "flow"
    df = df.iloc[1:, :]

    # Split the data into training and testing sets
    cutoff_index = int(len(df) * (1 - 0.3))
    train_data = df.iloc[:cutoff_index, :]
    test_data = df.iloc[cutoff_index:, :]

    # Define the input features and target variable
    input_features = ['flow_lagged']
    target = 'flow'

    # Fit a GradientBoostingRegressor to the training data
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    model.fit(train_data[input_features], train_data[target])

    # Use the model to make predictions on the testing data
    test_predictions = model.predict(test_data[input_features])

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_data[target], test_predictions)
    mse = mean_squared_error(test_data[target], test_predictions)
    rmse = mean_squared_error(test_data[target], test_predictions, squared=False)
    r2 = r2_score(test_data[target], test_predictions)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.2f}")


feature_importances = {'10d_cumulative_precipitation': 0.0153851716676228,
                         '10d_max_precipitation': 0.004812497265640948,
                         '10d_max_temperature': 0.0016156421660037802,
                         '10d_mean_precipitation': 0.035851756507517527,
                         '10d_mean_temperature': 0.1061839474265077,
                         '10d_min_temperature': 0.0004289494945303046,
                         '10d_std_precipitation': 0.001008531734665681,
                         '10d_std_temperature': 0.0009636175066720371,
                         '15d_cumulative_precipitation': 0.008435195045469395,
                         '15d_max_precipitation': 0.00035617006361715587,
                         '15d_max_temperature': 0.00017294661296650337,
                         '15d_mean_precipitation': 0.019702255563297688,
                         '15d_mean_temperature': 0.0028665503878601583,
                         '15d_min_temperature': 0.000445166508477506,
                         '15d_std_precipitation': 0.00018771338112327684,
                         '15d_std_temperature': 0.0005590803862567718,
                         '1d_max_precipitation': 0.00014137957966204416,
                         '1d_max_temperature': 0.0013272949644296437,
                         '1d_mean_precipitation': 0.0,
                         '1d_mean_temperature': 0.00824860579063988,
                         '1d_min_temperature': 0.0,
                         '20d_cumulative_precipitation': 0.0024615820219039096,
                         '20d_max_precipitation': 0.0004627903925256208,
                         '20d_max_temperature': 0.0,
                         '20d_mean_precipitation': 0.0024859189554157324,
                         '20d_mean_temperature': 0.0004501966704995562,
                         '20d_min_temperature': 0.003856341423744058,
                         '20d_std_precipitation': 0.002327328842262746,
                         '20d_std_temperature': 0.0020725409674659952,
                         '25d_cumulative_precipitation': 0.0013169269136133726,
                         '25d_max_precipitation': 0.00028797156243386054,
                         '25d_max_temperature': 0.002180578605952899,
                         '25d_mean_precipitation': 0.00868047796123978,
                         '25d_mean_temperature': 0.004203310368995262,
                         '25d_min_temperature': 1.4002523456011672e-06,
                         '25d_std_precipitation': 0.0004753309217353898,
                         '25d_std_temperature': 0.0005474271703486798,
                         '30d_cumulative_precipitation': 0.0035276710482657824,
                         '30d_delta_mean_temperature': 0.007794666670557412,
                         '30d_max_precipitation': 0.00039089380304688573,
                         '30d_max_temperature': 0.0020781991872786676,
                         '30d_mean_precipitation': 0.001977283022821122,
                         '30d_mean_temperature': 0.004288560594622804,
                         '30d_min_temperature': 0.004021832158801507,
                         '30d_std_precipitation': 0.00029895186686142633,
                         '30d_std_temperature': 0.004117211279584557,
                         '5d_cumulative_precipitation': 0.02458475364713526,
                         '5d_max_precipitation': 0.006513999559484964,
                         '5d_max_temperature': 0.005104506281637633,
                         '5d_mean_precipitation': 0.011010096357276972,
                         '5d_mean_temperature': 0.08564664999748955,
                         '5d_min_temperature': 0.019104691236693696,
                         '5d_std_precipitation': 0.002757789707126553,
                         '5d_std_temperature': 0.0003536242042187871,
                         'day_of_year_cos': 0.1373595079511596,
                         'interaction_p_0_t_0': 0.0,
                         'interaction_p_0_t_1': 0.0,
                         'interaction_p_0_t_2': 0.00020848158390714128,
                         'interaction_p_0_t_3': 0.0,
                         'interaction_p_0_t_4': 0.0,
                         'interaction_p_0_t_5': 1.56367987075738e-06,
                         'interaction_p_0_t_6': 0.0,
                         'interaction_p_0_t_7': 0.0006705060352669574,
                         'interaction_p_0_t_8': 1.6860020861072813e-08,
                         'interaction_p_1_t_0': 0.0003889133160492561,
                         'interaction_p_1_t_1': 0.0,
                         'interaction_p_1_t_2': 0.0,
                         'interaction_p_1_t_3': 0.0,
                         'interaction_p_1_t_4': 0.0,
                         'interaction_p_1_t_5': 0.00020345130038143368,
                         'interaction_p_1_t_6': 0.0,
                         'interaction_p_1_t_7': 3.6855821301326167e-09,
                         'interaction_p_1_t_8': 6.860484300209872e-06,
                         'interaction_p_2_t_0': 0.0,
                         'interaction_p_2_t_1': 0.0003721167467886544,
                         'interaction_p_2_t_2': 0.0007219150885415812,
                         'interaction_p_2_t_3': 0.0,
                         'interaction_p_2_t_4': 0.0056773708047516925,
                         'interaction_p_2_t_5': 0.002060515037067274,
                         'interaction_p_2_t_6': 0.0,
                         'interaction_p_2_t_7': 0.000691906587343248,
                         'interaction_p_2_t_8': 0.00494580937103157,
                         'interaction_p_3_t_0': 0.0,
                         'interaction_p_3_t_1': 0.0,
                         'interaction_p_3_t_2': 2.3896363567134844e-07,
                         'interaction_p_3_t_3': 0.0,
                         'interaction_p_3_t_4': 0.0,
                         'interaction_p_3_t_5': 0.0001686119037440398,
                         'interaction_p_3_t_6': 0.0,
                         'interaction_p_3_t_7': 0.0,
                         'interaction_p_3_t_8': 9.209204091190322e-07,
                         'interaction_p_4_t_0': 0.0,
                         'interaction_p_4_t_1': 0.0,
                         'interaction_p_4_t_2': 0.0,
                         'interaction_p_4_t_3': 0.0,
                         'interaction_p_4_t_4': 0.0,
                         'interaction_p_4_t_5': 5.39676646049059e-06,
                         'interaction_p_4_t_6': 0.0,
                         'interaction_p_4_t_7': 0.0,
                         'interaction_p_4_t_8': 0.0,
                         'interaction_p_5_t_0': 0.0,
                         'interaction_p_5_t_1': 0.0,
                         'interaction_p_5_t_2': 0.0,
                         'interaction_p_5_t_3': 0.0,
                         'interaction_p_5_t_4': 0.0,
                         'interaction_p_5_t_5': 0.0,
                         'interaction_p_5_t_6': 0.0,
                         'interaction_p_5_t_7': 0.0,
                         'interaction_p_5_t_8': 0.0003626683001049611,
                         'interaction_p_6_t_0': 0.0,
                         'interaction_p_6_t_1': 0.00467900897255631,
                         'interaction_p_6_t_2': 0.0,
                         'interaction_p_6_t_3': 0.0,
                         'interaction_p_6_t_4': 0.0,
                         'interaction_p_6_t_5': 2.5808063461973526e-06,
                         'interaction_p_6_t_6': 0.0,
                         'interaction_p_6_t_7': 0.0,
                         'interaction_p_6_t_8': 6.504941969899998e-05,
                         'interaction_p_7_t_0': 0.0,
                         'interaction_p_7_t_1': 0.0,
                         'interaction_p_7_t_2': 0.0,
                         'interaction_p_7_t_3': 0.00022575561259734905,
                         'interaction_p_7_t_4': 0.0034178090000486277,
                         'interaction_p_7_t_5': 0.0003029337631992616,
                         'interaction_p_7_t_6': 0.0,
                         'interaction_p_7_t_7': 0.0,
                         'interaction_p_7_t_8': 0.0,
                         'interaction_p_8_t_0': 8.31049210638488e-06,
                         'interaction_p_8_t_1': 0.0,
                         'interaction_p_8_t_2': 0.0,
                         'interaction_p_8_t_3': 0.0,
                         'interaction_p_8_t_4': 0.0006360521754901038,
                         'interaction_p_8_t_5': 0.0004229813135925835,
                         'interaction_p_8_t_6': 0.0,
                         'interaction_p_8_t_7': 0.0,
                         'interaction_p_8_t_8': 0.0,
                         'interaction_tdp_10d_mean_temperature': 0.0,
                         'interaction_tdp_15d_mean_temperature': 0.0,
                         'interaction_tdp_20d_mean_temperature': 0.0006659860593856406,
                         'interaction_tdp_25d_mean_temperature': 0.0,
                         'interaction_tdp_30d_mean_temperature': 4.5680020100768333e-07,
                         'interaction_tdp_5d_mean_temperature': 0.006732380277810662,
                         'lag(10d_cumulative_precipitation,1)': 0.0,
                         'lag(10d_cumulative_precipitation,2)': 0.0,
                         'lag(10d_cumulative_precipitation,3)': 0.0,
                         'lag(10d_cumulative_precipitation,4)': 1.9590371512377825e-05,
                         'lag(10d_cumulative_precipitation,5)': 0.00027776176319339706,
                         'lag(10d_max_precipitation,1)': 0.0,
                         'lag(10d_max_precipitation,2)': 0.0002856547627646467,
                         'lag(10d_max_precipitation,3)': 0.00033939436980405074,
                         'lag(10d_max_precipitation,4)': 0.00013453639045948643,
                         'lag(10d_max_temperature,1)': 0.0002345199943624964,
                         'lag(10d_max_temperature,2)': 4.466921805851213e-05,
                         'lag(10d_max_temperature,3)': 0.0,
                         'lag(10d_max_temperature,4)': 0.0,
                         'lag(10d_max_temperature,5)': 0.0,
                         'lag(10d_mean_precipitation,1)': 0.00018238169227315114,
                         'lag(10d_mean_precipitation,2)': 7.827637279663923e-05,
                         'lag(10d_mean_precipitation,3)': 0.0,
                         'lag(10d_mean_precipitation,4)': 6.838514502835333e-05,
                         'lag(10d_mean_precipitation,5)': 0.00028195796336028667,
                         'lag(10d_mean_temperature,1)': 0.00043993580821962824,
                         'lag(10d_mean_temperature,2)': 0.0021506177203128225,
                         'lag(10d_mean_temperature,3)': 0.0006130096309974079,
                         'lag(10d_mean_temperature,4)': 0.0,
                         'lag(10d_mean_temperature,5)': 0.0005162194704007456,
                         'lag(10d_min_temperature,1)': 0.0,
                         'lag(10d_min_temperature,2)': 4.515602515086288e-05,
                         'lag(10d_min_temperature,3)': 3.833377352358481e-05,
                         'lag(10d_min_temperature,4)': 0.0,
                         'lag(10d_min_temperature,5)': 0.00045507449147798865,
                         'lag(10d_std_precipitation,1)': 0.0,
                         'lag(10d_std_precipitation,2)': 0.00012066190198466413,
                         'lag(10d_std_precipitation,3)': 0.0,
                         'lag(10d_std_precipitation,4)': 0.0005012474343761175,
                         'lag(10d_std_temperature,1)': 0.0001259380760943449,
                         'lag(10d_std_temperature,2)': 0.0,
                         'lag(10d_std_temperature,3)': 0.0002340956369116097,
                         'lag(10d_std_temperature,4)': 0.0003387888312929231,
                         'lag(10d_std_temperature,5)': 7.72068429612144e-05,
                         'lag(15d_cumulative_precipitation,1)': 0.0004578062990036661,
                         'lag(15d_cumulative_precipitation,2)': 0.0,
                         'lag(15d_cumulative_precipitation,3)': 0.0,
                         'lag(15d_cumulative_precipitation,4)': 0.0007030619672592722,
                         'lag(15d_max_precipitation,1)': 0.0003525983830302649,
                         'lag(15d_max_precipitation,2)': 0.00030097638799676595,
                         'lag(15d_max_precipitation,3)': 0.0001742068138297638,
                         'lag(15d_max_temperature,1)': 0.0,
                         'lag(15d_max_temperature,2)': 0.0003268824484753909,
                         'lag(15d_max_temperature,3)': 0.00021885700453552932,
                         'lag(15d_max_temperature,4)': 0.0,
                         'lag(15d_max_temperature,5)': 0.0004893706128220944,
                         'lag(15d_mean_precipitation,1)': 0.00038418026449206635,
                         'lag(15d_mean_precipitation,2)': 0.0003003682408742787,
                         'lag(15d_mean_precipitation,3)': 6.137054961222695e-05,
                         'lag(15d_mean_precipitation,4)': 0.0,
                         'lag(15d_mean_temperature,1)': 0.002475133278961291,
                         'lag(15d_mean_temperature,2)': 0.0,
                         'lag(15d_mean_temperature,3)': 0.0,
                         'lag(15d_mean_temperature,4)': 0.0,
                         'lag(15d_mean_temperature,5)': 0.0,
                         'lag(15d_min_temperature,1)': 0.0012136007699965715,
                         'lag(15d_min_temperature,2)': 0.0002696181699992881,
                         'lag(15d_min_temperature,3)': 0.0006445158926439155,
                         'lag(15d_min_temperature,4)': 0.00038900870725377696,
                         'lag(15d_min_temperature,5)': 0.004087567881244811,
                         'lag(15d_std_precipitation,1)': 0.0009179794844363658,
                         'lag(15d_std_precipitation,2)': 1.540405831121625e-05,
                         'lag(15d_std_precipitation,3)': 0.0,
                         'lag(15d_std_precipitation,4)': 1.92418498623184e-05,
                         'lag(15d_std_temperature,1)': 0.0005519570347669705,
                         'lag(15d_std_temperature,2)': 5.212182289278459e-05,
                         'lag(15d_std_temperature,3)': 0.000341597112115072,
                         'lag(15d_std_temperature,4)': 0.00016566737984432107,
                         'lag(15d_std_temperature,5)': 0.0,
                         'lag(20d_cumulative_precipitation,1)': 0.00011427179534818453,
                         'lag(20d_cumulative_precipitation,2)': 0.0,
                         'lag(20d_cumulative_precipitation,3)': 0.0,
                         'lag(20d_max_precipitation,1)': 0.0005191797573027856,
                         'lag(20d_max_precipitation,2)': 8.570796955002288e-05,
                         'lag(20d_max_temperature,1)': 0.00013926321395076022,
                         'lag(20d_max_temperature,2)': 0.001197311706633701,
                         'lag(20d_max_temperature,3)': 0.000367361271149799,
                         'lag(20d_max_temperature,4)': 0.00030913413341278865,
                         'lag(20d_max_temperature,5)': 0.0006408466595391684,
                         'lag(20d_mean_precipitation,1)': 0.0016109619020057597,
                         'lag(20d_mean_precipitation,2)': 7.791710832082623e-07,
                         'lag(20d_mean_precipitation,3)': 0.00011971798097536641,
                         'lag(20d_mean_temperature,1)': 0.0,
                         'lag(20d_mean_temperature,2)': 8.021575915870577e-08,
                         'lag(20d_mean_temperature,3)': 0.0,
                         'lag(20d_mean_temperature,4)': 0.0009952741169858158,
                         'lag(20d_mean_temperature,5)': 0.0011910092532230144,
                         'lag(20d_min_temperature,1)': 0.002799442706422159,
                         'lag(20d_min_temperature,2)': 0.0007577715023337176,
                         'lag(20d_min_temperature,3)': 0.0005703822837341354,
                         'lag(20d_min_temperature,4)': 4.550101395222495e-09,
                         'lag(20d_min_temperature,5)': 0.000407427666014322,
                         'lag(20d_std_precipitation,1)': 0.00038769294884609065,
                         'lag(20d_std_precipitation,2)': 0.0,
                         'lag(20d_std_precipitation,3)': 0.0014428083725612662,
                         'lag(20d_std_temperature,1)': 0.00030234961142961916,
                         'lag(20d_std_temperature,2)': 0.0,
                         'lag(20d_std_temperature,3)': 0.0,
                         'lag(20d_std_temperature,4)': 4.036071751689749e-05,
                         'lag(20d_std_temperature,5)': 0.0002743450159016469,
                         'lag(25d_cumulative_precipitation,1)': 0.0002867713155234778,
                         'lag(25d_cumulative_precipitation,2)': 0.0,
                         'lag(25d_max_precipitation,1)': 4.590227918359644e-06,
                         'lag(25d_max_temperature,1)': 0.0,
                         'lag(25d_max_temperature,2)': 0.0013601821797695846,
                         'lag(25d_max_temperature,3)': 0.0,
                         'lag(25d_max_temperature,4)': 0.008600211364483159,
                         'lag(25d_mean_precipitation,1)': 0.0010224821233688188,
                         'lag(25d_mean_precipitation,2)': 0.00031122204416834855,
                         'lag(25d_mean_temperature,1)': 0.0002952343107818178,
                         'lag(25d_mean_temperature,2)': 0.0,
                         'lag(25d_mean_temperature,3)': 0.0,
                         'lag(25d_mean_temperature,4)': 0.0004756354893592435,
                         'lag(25d_mean_temperature,5)': 0.0046251975385223215,
                         'lag(25d_min_temperature,1)': 0.0023357255243926975,
                         'lag(25d_min_temperature,2)': 0.00013341219626415593,
                         'lag(25d_min_temperature,3)': 0.0038838748522262373,
                         'lag(25d_min_temperature,4)': 0.000748139194511195,
                         'lag(25d_min_temperature,5)': 0.0017241747582643129,
                         'lag(25d_std_precipitation,1)': 0.0,
                         'lag(25d_std_precipitation,2)': 0.0005078004716250463,
                         'lag(25d_std_temperature,1)': 6.245293829640145e-05,
                         'lag(25d_std_temperature,2)': 0.0007820384766942383,
                         'lag(25d_std_temperature,3)': 0.00022737657352609153,
                         'lag(25d_std_temperature,4)': 0.0,
                         'lag(25d_std_temperature,5)': 0.000337458625449759,
                         'lag(30d_cumulative_precipitation,1)': 0.00045821709741275074,
                         'lag(30d_max_temperature,1)': 0.0006159761944555776,
                         'lag(30d_max_temperature,2)': 0.005046517001174908,
                         'lag(30d_max_temperature,3)': 0.028704762039271407,
                         'lag(30d_mean_precipitation,1)': 0.000554351434813479,
                         'lag(30d_mean_temperature,1)': 0.0,
                         'lag(30d_mean_temperature,2)': 0.0,
                         'lag(30d_mean_temperature,3)': 0.0019001638693703102,
                         'lag(30d_mean_temperature,4)': 0.004178787858877647,
                         'lag(30d_mean_temperature,5)': 0.17682535584559947,
                         'lag(30d_min_temperature,1)': 0.0,
                         'lag(30d_min_temperature,2)': 0.0014031523655817087,
                         'lag(30d_min_temperature,3)': 0.0,
                         'lag(30d_min_temperature,4)': 0.006027254708653782,
                         'lag(30d_min_temperature,5)': 0.0031272459726967433,
                         'lag(30d_std_precipitation,1)': 0.00045205581623752384,
                         'lag(30d_std_temperature,1)': 4.275868045667285e-05,
                         'lag(30d_std_temperature,2)': 0.00012017662891864901,
                         'lag(30d_std_temperature,3)': 0.0005016087063102928,
                         'lag(30d_std_temperature,4)': 0.000675624909870471,
                         'lag(30d_std_temperature,5)': 4.6456296975361055e-05,
                         'lag(5d_cumulative_precipitation,1)': 0.0012783222183807483,
                         'lag(5d_cumulative_precipitation,2)': 0.0,
                         'lag(5d_cumulative_precipitation,3)': 9.434537748327179e-05,
                         'lag(5d_cumulative_precipitation,4)': 0.0002476869778046523,
                         'lag(5d_cumulative_precipitation,5)': 9.84473267642307e-05,
                         'lag(5d_max_precipitation,1)': 0.0017180089311650934,
                         'lag(5d_max_precipitation,2)': 7.246639548509097e-05,
                         'lag(5d_max_precipitation,3)': 0.00010487736965969419,
                         'lag(5d_max_precipitation,4)': 0.0,
                         'lag(5d_max_temperature,1)': 0.00030250688603631595,
                         'lag(5d_max_temperature,2)': 0.0002004566065511525,
                         'lag(5d_max_temperature,3)': 0.0,
                         'lag(5d_max_temperature,4)': 0.0,
                         'lag(5d_max_temperature,5)': 0.0,
                         'lag(5d_mean_precipitation,1)': 0.0011004561589226778,
                         'lag(5d_mean_precipitation,2)': 0.0,
                         'lag(5d_mean_precipitation,3)': 0.0,
                         'lag(5d_mean_precipitation,4)': 3.138386551622707e-05,
                         'lag(5d_mean_precipitation,5)': 0.0,
                         'lag(5d_mean_temperature,1)': 0.0002414294538486573,
                         'lag(5d_mean_temperature,2)': 0.0020062922040095695,
                         'lag(5d_mean_temperature,3)': 0.0,
                         'lag(5d_mean_temperature,4)': 1.3361000582773313e-05,
                         'lag(5d_mean_temperature,5)': 0.00018100490506518254,
                         'lag(5d_min_temperature,1)': 0.0010546928012132644,
                         'lag(5d_min_temperature,2)': 0.0,
                         'lag(5d_min_temperature,3)': 5.025038834674855e-05,
                         'lag(5d_min_temperature,4)': 0.00011236558890222425,
                         'lag(5d_min_temperature,5)': 0.0,
                         'lag(5d_std_precipitation,1)': 5.9612098852572406e-05,
                         'lag(5d_std_precipitation,2)': 0.0001294203248225575,
                         'lag(5d_std_precipitation,3)': 0.0006172281207086068,
                         'lag(5d_std_precipitation,4)': 0.0,
                         'lag(5d_std_temperature,1)': 7.917575710800106e-05,
                         'lag(5d_std_temperature,2)': 0.0001374744730974265,
                         'lag(5d_std_temperature,3)': 0.00025462402932453207,
                         'lag(5d_std_temperature,4)': 0.00011813158767907599,
                         'lag(5d_std_temperature,5)': 8.528131067797045e-05,
                         'month_cos': 0.026017598492661914,
                         'p_0': 0.0,
                         'p_1': 2.56962220114272e-05,
                         'p_2': 0.00012528819112015603,
                         'p_3': 0.0002206569851267948,
                         'p_4': 7.537343332219571e-05,
                         'p_5': 0.0,
                         'p_6': 0.0006487835004842718,
                         'p_7': 0.0,
                         't_0': 0.0,
                         't_1': 0.0,
                         't_2': 0.0018545470508495798,
                         't_3': 0.0,
                         't_4': 0.0047940961827050085,
                         't_5': 0.0004511040613786414,
                         't_6': 0.00021381501690506736,
                         't_7': 5.610369518587507e-05,
                         't_8': 0.06469342335527796,
                         'tdp': 0.00015971961398600834}


def get_top_n_features(feature_importances, n):
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_n_features = [feature[0] for feature in sorted_features[:n]]
    return top_n_features


x = get_top_n_features(feature_importances, 50)

test_model(df)