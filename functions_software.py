import numpy as np
import scipy.signal as ss
import statsmodels.api as sm
from scipy.integrate import cumulative_trapezoid
import pywt
import matplotlib.pyplot as plt

# bandpass filtering
def bandpass(phase_diff, waveform_output, filter_order = 7):
    # funtion takes the phase difference and desired filtered waveform, returns filtered phase with bandpass method
    # default filter order of 7
    bpf = [0.1, 0.6, 0.6, 4]  # Bandpass filter cutoff frequencies -> [RR_low, RR_high, HR_low, HR_high]
    Fs = 10  # sampling rate, Hz

    # breathing waveform
    if waveform_output == 'RR':
        # design filter
        b, a = ss.butter(filter_order, [bpf[0], bpf[1]], btype='bandpass', fs = Fs)
        # apply filter
        wf_filt = ss.lfilter(b, a, phase_diff)
        
        return wf_filt
    # heartbeat waveform 
    elif waveform_output == 'HR':
        b, a = ss.butter(filter_order, [bpf[0], bpf[1]], btype='bandpass', fs = Fs)
        wf_filt = ss.lfilter(b, a, phase_diff)

        return wf_filt
    else:
        print('Unrecognized desired waveform! Passing options are "HR" and "RR"') 

# function finds the median frequency given fft frequency (frq) and amplitude (Y) data 
def fft_median(Y, frq):
    # find area under curve using cumulative integration using trapezoil rule 
    area = cumulative_trapezoid(Y, frq)
    area = area / area[-1]
    idx = np.argwhere(area >= .5)[0,0]  # return the index where the area is at least 50%
    rate = 60*frq[idx]

    return rate

# function performs fft and returns RR, HR from previous 10 seconds of data
def perform_fft(phase_unwrap):
    # need this line below: (suuuper unfortunate function name for cumuative integration using the trapezoid rule)
    #from scipy.integrate import cumtrapz

    Fs = 10  # sampling rate, Hz
    delta_t_single = float(1/Fs)
    bin_size_samples = len(phase_unwrap)
    bpf = [0.1, 0.6, 0.6, 4]  # Bandpass filter cutoff frequencies -> [RR_low, RR_high, HR_low, HR_high]
    filter_order = 7
    HR_fft_est = []

    for i in range(len(phase_unwrap)):
        phase_unwrap[i] = phase_unwrap[i].replace('"(', '')
        phase_unwrap[i] = phase_unwrap[i].replace(',)"', '')

    # take the difference of successive phase measurments
    phase = [b - a for a, b in zip(phase_unwrap, phase_unwrap[1:])]
    
    ######
    # HR #
    ######
    #filter HR
    b, a = ss.butter(filter_order, [bpf[2], bpf[3]], btype='bandpass', fs = Fs)
    HR_filt = ss.lfilter(b, a, phase)

    # fft
    n = len(HR_filt)
    hann_window = np.fft.rfft(HR_filt * ss.windows.hann(n, sym=False), n=n)
    frq = np.fft.rfftfreq(n, d=delta_t_single)  # x data, 0:0.1:5
    Y = 2.0/bin_size_samples*np.abs(hann_window)    # y data

    # find area under curve using cumulative integration using trapezoil rule 
    area = cumtrapz(Y, frq)
    area = area / area[-1]
    idx = np.argwhere(area >= .5)[0,0]  # return the index where the area is at least 50%

    HR_fft_est = 60*frq[idx]
    
    ######
    # RR #
    ######
    # filter RR
    b, a = ss.butter(filter_order, [bpf[0], bpf[1]], btype='bandpass', fs = Fs)
    RR_filt = ss.lfilter(b, a, phase)
    
    # fft
    n = len(RR_filt)
    hann_window = np.fft.rfft(HR_filt * ss.windows.hann(n, sym=False), n=n)
    frq = np.fft.rfftfreq(n, d=delta_t_single)  # x data, 0:0.1:5
    Y = 2.0/bin_size_samples*np.abs(hann_window)    # y data

    # find area under curve using cumulative integration using trapezoil rule 
    area = cumtrapz(Y, frq)
    area = area / area[-1]
    idx = np.argwhere(area >= .5)[0,0]  # return the index where the area is at least 50%

    RR_fft_est = 60*frq[idx]

    return RR_fft_est, HR_fft_est

# function performs fft and returns RR, HR from previous 10 seconds of data
def perform_fft_new(phase_unwrap, desired_waveform):
    Fs = 10  # sampling rate, Hz
    delta_t_single = float(1/Fs)
    bin_size_samples = len(phase_unwrap)

    phase = phase_diff(phase_unwrap)
    filt = bandpass(phase, desired_waveform)

    # fft
    n = len(filt)
    hann_window = np.fft.rfft(filt * ss.windows.hann(n, sym=False), n=n)
    frq = np.fft.rfftfreq(n, d=delta_t_single)  # x data, 0:0.1:5
    Y = 2.0/bin_size_samples*np.abs(hann_window)    # y data
    
    return Y, frq

# phase difference
def phase_diff(phase_unwrap):
    # pass through unwrapped phase, this is the measurement from the sensor
    phase_diff_val = []  # function returns the successive phase differences

    for a, b in zip(phase_unwrap, phase_unwrap[1:]):
        if b-a > np.pi:
            phase_diff_val.append(b-a-np.pi)
        elif b-a < -np.pi:
            phase_diff_val.append(b-a+np.pi)
        else:
            phase_diff_val.append(b-a)

    return phase_diff_val

# Function to plot the detail coefficient stuff from https://notebook.community/CSchoel/learn-wavelets/wavelet-denoising
def plot_dwt(details, approx, xlim=(-300,300), **line_kwargs):
    for i in range(len(details)):
        plt.subplot(len(details)+1,1,i+1)
        d = details[len(details)-1-i]
        half = len(d)//2
        xvals = np.arange(-half,-half+len(d))* 2**i
        plt.plot(xvals, d, **line_kwargs)
        # coefficient peaks
        peaks,_ = ss.find_peaks(d)
        print('Detail Coefficient %g has %g peaks or %5.2f bpm' % (i, len(peaks), 60*len(peaks)/(time_[-1] - time_[0])))
        #print('Detail Coefficient %g has %g peaks' % (i, len(peaks)))
        plt.scatter(x=xvals[peaks], y=d[peaks], c='r')
        plt.xlim(xlim)
        plt.title("detail[{}]".format(i))
    plt.subplot(len(details)+1,1,len(details)+1)
    plt.title("approx")
    plt.plot(xvals, approx, **line_kwargs)
    plt.xlim(xlim)
 
# function that performs autocorrelation and raturns RR, HR from 30 seconds of data
def process_auto(phase_unwrap):
    # given a set of data (for best values, should be about 30 seconds of data)
    # perform autocorrelation and find top 3 peaks 
    # find rate based on peak values
    # return a HR and RR

    # need numpy as np, scipy.signal as ss, and statsmodels.api as sm

    # if phase format is "(__,)", comment out if not
    for i in range(len(phase_unwrap)):
        phase_unwrap[i] = phase_unwrap[i].replace('"(', '')
        phase_unwrap[i] = phase_unwrap[i].replace(',)"', '')

    # take the difference of successive phase measurments
    phase = [b - a for a, b in zip(phase_unwrap, phase_unwrap[1:])]

    Fs = 10  # sampling rate, Hz
    delta_t_single = float(1/Fs)
    bpf = [0.1, 0.6, 0.6, 4]  # Bandpass filter cutoff frequencies -> [RR_low, RR_high, HR_low, HR_high]
    filter_order = 7
    peak_count_RR_auto = 3
    peak_count_HR_auto = 15

    ######
    # HR #
    ######
    #filter HR
    b, a = ss.butter(filter_order, [bpf[2], bpf[3]], btype='bandpass', fs = Fs)
    HR_filt = ss.lfilter(b, a, phase)
    # autocorrelation
    acf_HR = sm.tsa.acf(HR_filt, nlags=(len(HR_filt) -1))
    # HR: find peaks in autocorrelation
    peaks_HR,_ = ss.find_peaks(acf_HR, 0)  # returns index of acf_HR peaks greater than zero
    # if there are at least peak_count peaks, calculate rate
    if len(peaks_HR) > peak_count_HR_auto:
        peaks_HR_1 = np.concatenate(([0], peaks_HR[:peak_count_HR_auto-1]), axis=0)
        peaks_HR_2 = peaks_HR[:peak_count_HR_auto]
        delta_t_HR = (peaks_HR_2 - peaks_HR_1)*delta_t_single
        mean_delta_t_HR = np.mean(delta_t_HR)
        mean_HR_auto = 60/mean_delta_t_HR
    # if no peaks, no data
    else:
        mean_HR_auto = '' 
        print('not enough HR ACF peaks: ', len(peaks_HR))

    ######
    # RR #
    ######
    # filter RR
    b, a = ss.butter(filter_order, [bpf[0], bpf[1]], btype='bandpass', fs = Fs)
    RR_filt = ss.lfilter(b, a, phase)
    # autocorrelation
    acf_RR = sm.tsa.acf(RR_filt, nlags=(len(RR_filt) -1))    # need larger bin sizes to get multiple RR autocorr peaks
    # find peaks in autocorrelation and calcualte average
    peaks_RR,_ = ss.find_peaks(acf_RR, 0)
    # if there are enough peaks to estimate, calculate rate
    if len(peaks_RR) > peak_count_RR_auto:
        peaks_RR_1 = np.concatenate(([0], peaks_RR[:peak_count_RR_auto-1]), axis=0)
        peaks_RR_2 = peaks_RR[:peak_count_RR_auto]
        delta_t_RR = (peaks_RR_2 - peaks_RR_1)*delta_t_single
        mean_delta_t_RR = np.mean(delta_t_RR)
        mean_RR_auto = 60/mean_delta_t_RR
    else:
        mean_RR_auto = ''
        print('not enough RR ACF peaks: ', len(peaks_RR))

    return mean_RR_auto, mean_HR_auto

# wavelet coefficient thresholding from https://notebook.community/CSchoel/learn-wavelets/wavelet-denoising
def thresh(coeffs, waveform_output):
    # funtion takes wavelet coefficients from deconstruction and calculates thresholding values
    # from research, we know to do this at and above the level of 4 for HR and 6 for RR
    
    # returns thresholded coefficients to be used for reconstruction
    thresh = []
    i=0

    # set level
    if waveform_output == 'HR':
        level = 4
    elif waveform_output == 'RR':
        level = 6
    else:
        print('Unrecognized desired waveform! Passing options are "HR" and "RR"')

    for c in coeffs:
        # coeffs: an, dn, dn-1, dn-2, ... , d1
        # coeff_level: n, n, n-1, n-2, ... , 1
        coeff_level = np.append(len(coeffs)-1, np.linspace(len(coeffs)-1, 1, num=len(coeffs)-1))
        
        if coeff_level[i] > level:  # for coeff levels above decomp level, calculate threshold
            n = np.size(c)
            a = np.sort(np.abs(c))**2
            b = np.linspace(n-1,0,n)
            s = np.cumsum(a)+b*a
            risk = (n - (2 * np.arange(n)) + s)/n
            ibest = np.argmin(risk)
            thresh.append(np.sqrt(a[ibest]))
        else:   # else, set a threshold of zero so levels are not filtered
            thresh.append(20)    # with soft thresholding, coefficients less than the threshold are chosen, others are set to zero
        i = i+1

    return thresh

# read in data from text file and output phase difference
def txt_file_read(file_name, sensor_vals=False):
    # returns time, phase_diff by defualt
    # if sensor_vals==True, the function returns time, phase_diff, HR_sensor, and RR_sensor
    phase_unwrap = [];
    HR_sensor = [];
    time = [];
    RR_sensor = [];

    # file_name example: 'mmwave_data_1_1_2024.txt' (use quotes)
    # read in the data file, skill the column titles, return time, HR from sensor, RR from sensor, unwrapped phase 
    t, h, r, p, _, _ = np.loadtxt('C:\\Users\\zkaya\\OneDrive - The Ohio State University\\Python3\\mmwave data\\'+file_name, dtype='str' ,skiprows=1, unpack=True)

    # trim parentheses and commas from data
    for i in range(len(t)):
        h[i] = h[i].replace('"(', '')
        h[i] = h[i].replace(',)"', '')
        r[i] = r[i].replace('"(', '')
        r[i] = r[i].replace(',)"', '')
        p[i] = p[i].replace('"(', '')
        p[i] = p[i].replace(',)"', '')
        phase_unwrap.append(float(p[i]))
        time.append(float(t[i]))
        HR_sensor.append(float(h[i]))
        RR_sensor.append(float(r[i]))
    
    if sensor_vals == True:
        return time, phase_unwrap, HR_sensor, RR_sensor
    elif sensor_vals == False:
        return time, phase_unwrap
    
# wavelet filtering output thresholded coeffs
def wavelet_filter(phase_unwrap, waveform_output, wavelet_deconstruct_level=7):
    # funtion takes the unwrapped phase measurement from the sensor and desired waveform and filters it with wavelet method
    # default wavelet deconstruction is at level 7, per research papers
    # deconstruction is also done with Debauchies 5 
    # returns the waveform 

    phase_diff_val = phase_diff(phase_unwrap)
    # deconstruction with wavelet
    coeffs = pywt.wavedec(phase_diff_val, 'db5', level=wavelet_deconstruct_level)
    # coefficient thresholding
    thresholded_coeffs = [pywt.threshold(c, t, 'soft') for c, t in zip(coeffs, thresh(coeffs, waveform_output))]
    # wavelet reconstruction
    waveform = pywt.waverec(thresholded_coeffs, 'db5')

    return waveform

# RR estimate from waveform using peaks
def wf_to_rate(waveform, pk_prominence=0.1, pk_width=10):
    # function finds the period in the waveform and converts to rate/min
    # inputs:
    #   waveform: at least 5 seconds of data for RR estimation
    #   pk_prominence: 0.1 unless otherwise specified, the vertical height difference between peak found and surrounding data points
    #   pk_width: 10 unless otherwise specified, the horizontal width of a peak in number of data points
    #       peaks of waveform should have some width, for a 10 Hz sampling rate, this is a width of 1 second which is reasonable
    # returns rate in beats or breaths per minute
    
    delta_t_single = 0.1
    time_wf = []

    wf_peaks,_ = ss.find_peaks(waveform, prominence=pk_prominence, width=pk_width)
    wf_peak_diff = np.diff(wf_peaks)
    # caclulate rate, convert index difference (i_d) to time difference (10 Hz/i_d), convert time difference to bpm (*60 seconds/min)
    rate_wf = 60/(wf_peak_diff*delta_t_single)
    
    # calculate time at calculated rate, convert index to time (10 Hz/i_d)
    # (for plotting)
    for peak in wf_peaks[:-1]:
        time_wf.append(peak*delta_t_single)
    
    return rate_wf, time_wf
 
