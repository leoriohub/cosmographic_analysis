import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(data_h0: list, data_q0: list, filename = None, titlemarker = 'ISO'):
    """
        Plots histograms for delta_h0_max and delta_q0_max.

        Parameters:
            - data_h0 (list): A list containing the h0 data and the data's max isotropy value.
            - data_q0 (list): A list containing the q0 data and the data's max isotropy value.
            - x_gauss (list): A list of x values for the Gaussian fit to be plotted.
            - y_gauss (list): A list of y values for the Gaussian fit to be plotted.
            - filename (str, optional): The name of the file to save the plot. Defaults to None.
            - titlemarker (str, optional): The title marker for the plot. Defaults to 'ISO'.

        Returns:
            None
    """

    delta_h0_max = data_h0[0]
    delta_q0_max = data_q0[0]
    delta_h0_data_max = data_h0[1]
    delta_q0_data_max = data_q0[1]
    x_gauss_h0 = data_h0[2]
    y_gauss_h0 = data_h0[3]
    x_gauss_q0 = data_q0[2]
    y_gauss_q0 = data_q0[3]

    percentile_5_h0 = np.percentile(delta_h0_max, 5)
    percentile_95_h0 = np.percentile(delta_h0_max, 95)
    percentile_5_q0 = np.percentile(delta_q0_max, 5)
    percentile_95_q0 = np.percentile(delta_q0_max, 95)
    
    hist_color = 'tab:blue'
    if titlemarker == 'LCDM':
        hist_color = 'pink'

    # Create h0 histogram using plt.subplots 
    hist4, ax4 = plt.subplots(1, 2, figsize=(15, 7))

    # Histogram for delta_h0_max
    counts_h0, bins_h0, _ = ax4[0].hist(
        delta_h0_max, bins='auto', density=False, alpha=0.7, color=hist_color)

    # Set the bin width to the difference between the first and second bin
    bin_width_h0 = bins_h0[1] - bins_h0[0]

    # Red line for data max anisotropy
    ax4[0].axvline(delta_h0_data_max, color='red', linestyle='solid',
                linewidth=3, label='data=%0.4f' % delta_h0_data_max)

    # Dashed lines  for 5% and 95% percentiles
    ax4[0].axvline(percentile_5_h0, color='black',
                linewidth=2, linestyle='dashed')

    ax4[0].axvline(percentile_95_h0, color='black',
                linewidth=2, linestyle='dashed')

    ax4[0].text(percentile_5_h0 - 2.2*bin_width_h0, 55, '5%', fontsize=20)

    ax4[0].text(percentile_95_h0+0.6*bin_width_h0, 55, '95%', fontsize=20)

    # Plot the Gaussian
    ax4[0].plot(x_gauss_h0, y_gauss_h0 * len(delta_h0_max) * bin_width_h0,
                'black', linewidth=2)  # Scale Gaussian fit to match total counts

    # Format x axis ticks with 3 decimals
    ax4[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Naming labels and titles
    ax4[0].set_xlabel(r'$Δh_0^{max}$')
    ax4[0].set_ylabel('Counts')
    ax4[0].legend(loc='upper right')
    ax4[0].set_title(f'$Δh_0^{{max}}$ {titlemarker}-Realizations')

    

    # Histogram for delta_q0_max
    counts_q0, bins_q0, _ = ax4[1].hist(
        delta_q0_max, bins='auto', density=False, alpha=0.7, color=hist_color)

    bin_width_q0 = bins_q0[1] - bins_q0[0]

    # Red line for data max anisotropy
    ax4[1].axvline(delta_q0_data_max, color='red', linestyle='solid',
                linewidth=3, label='data=%0.4f' % delta_q0_data_max)

    # Dashed lines  for 5% and 95% percentiles
    ax4[1].axvline(percentile_5_q0, color='black',
                linewidth=2, linestyle='dashed')

    ax4[1].axvline(percentile_95_q0, color='black',
                linewidth=2, linestyle='dashed')

    ax4[1].text(percentile_5_q0 - 1.8*bin_width_q0, 55, '5%', fontsize=20)

    ax4[1].text(percentile_95_q0 + 0.6 *
                bin_width_q0, 55, '95%', fontsize=20)


    # Plot the Gaussian
    ax4[1].plot(x_gauss_q0, y_gauss_q0 * len(delta_q0_max) * bin_width_q0,
                'black', linewidth=2)  # Scale Gaussian fit to match total counts

    ax4[1].set_xlabel(r'$Δq_0^{max}$')
    ax4[1].set_ylabel('Counts')
    ax4[1].legend(loc='upper right')
    ax4[1].set_title(f'$Δq_0^{{max}}$ {titlemarker}-Realizations')
    plt.tight_layout()
    if filename != None :
        plt.savefig(f'histograms/{filename}')
    plt.show()

    
def plot_both_histograms(data_h0: list, data_q0: list, filename = None) -> None:
    """
    This function plots both LCDM and ISO histograms in the same figure. It takes in two arrays, `data_h0` and `data_q0`. The function also accepts an optional `filename` parameter to save the plot as an image file.

    Parameters:
        - data_h0 (np.ndarray): An array containing the maximum h0 anisotropy for ISO, LCDM and the original data.
        - data_q0 (np.ndarray): An array containing the maximum q0 anisotropy for ISO, LCDM and the original data.
        - filename (str, optional): The name of the file to save the plot as. Defaults to None.

    Returns:
        None
    """

   
    delta_h0_iso_max = data_h0[0]
    delta_q0_iso_max = data_q0[0]
    delta_h0_lcdm_max = data_h0[1]
    delta_q0_lcdm_max = data_q0[1]
    delta_h0_data_max = data_h0[2]
    delta_q0_data_max = data_q0[2]
   
    delta_h0_lcdm_max = np.array(delta_h0_lcdm_max)
    data_q0_lcdm = np.array(delta_q0_lcdm_max)
    data_h0_iso = np.array(delta_h0_iso_max)
    data_q0_iso = np.array(delta_q0_iso_max)

    # Create the histograms with subplots
    hist2, ax2 = plt.subplots(1, 2, figsize=(15, 7))
    # Histogram for delta_h0 with different colors for LCDM and ISO
    ax2[0].hist(delta_h0_lcdm_max, color='pink',
            bins='auto', alpha=0.7, label='LCDM')
    ax2[0].hist(data_h0_iso, color='skyblue', bins='auto', alpha=0.7, label='ISO')
    ax2[0].axvline(delta_h0_data_max, color='red', linestyle='solid',
                linewidth=2, label='data=%0.4f' % delta_h0_data_max)
    ax2[0].legend(loc='upper right')  # Adjust the legend to the upper right
    ax2[0].set_xlabel(r'$Δh_0^{max}$')
    ax2[0].set_ylabel('Counts')

    # Histogram for delta_q0 with different colors for LCDM and ISO
    ax2[1].hist(data_q0_lcdm, color='pink', bins='auto', alpha=0.7, label='LCDM')
    ax2[1].hist(data_q0_iso, color='skyblue', bins='auto', alpha=0.7, label='ISO')
    ax2[1].axvline(delta_q0_data_max, color='red', linestyle='solid', linewidth=2,
                label='data=%0.4f' % delta_q0_data_max)  # plot the data value
    ax2[1].legend(loc='upper right')  # Adjust the legend to the upper right
    ax2[1].set_xlabel(r'$Δq_0^{max}$')
    ax2[1].set_ylabel('Counts')

    # Set the title and adjust the layout
    # hist2.suptitle(rf'{prefix_name} $Δh_{{\mathrm{{max}}}}$ and $Δq_{{\mathrm{{max}}}}$ histograms for LCDM and ISO distributions')
    plt.tight_layout()
    
    if filename != None :
        plt.savefig(f'histograms/{filename}')
    plt.show()   
