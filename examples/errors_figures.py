from quantification.metrics.binary import absolute_error, relative_absolute_error
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_results(path, pdf_path, norm, y_max='fixed'):

    pp = PdfPages(pdf_path)
    classes = os.listdir(path)
    preds = {}
    
    for cls in classes:
        preds[os.path.splitext(cls)[0]] = pd.read_csv(path + cls, index_col=0)
    
    classes = [os.path.splitext(cls)[0] for cls in classes]
    
    
    methods = ['CC', 'AC', 'PCC', 'PAC', 'HDy']
    
    # Normalizar
    if norm:
        for method in methods:
            all_preds = np.asarray([pred[method].values for pred in preds.values()])
            sum_preds = all_preds.sum(axis=0)
            for cls in preds:
                preds[cls][method] /= sum_preds
    
    errors = pd.DataFrame(columns=methods + ['Class'])
    r_errors = pd.DataFrame(columns=methods + ['Class'])
    true_negatives = pd.DataFrame(columns=methods, index=classes)
    
    for cls, pred in preds.iteritems():
        errors_ = pd.DataFrame(columns=methods + ['Class'])
        r_errors_ = pd.DataFrame(columns=methods + ['Class'])
        for method in methods:
            errors_[method] = absolute_error(pred['True'], pred[method])
            r_errors_[method] = relative_absolute_error(pred['True'], pred[method], epsilon=1/(2*3600.0))
        errors_['Class'] = cls
        r_errors_['Class'] = cls
        errors = errors.append(errors_)
        r_errors = r_errors.append(r_errors_)
        
        idx = (pred['True'] <= 0.001)
        true_zeros = idx.sum()
        for method in methods:
            if true_zeros:
                true_negatives.loc[cls,method] = (pred[method][idx] <= 0.001).sum() / float(true_zeros)
            else:
                true_negatives.loc[cls,method] = np.nan
    means = errors.groupby('Class').mean()
    r_means = r_errors.groupby('Class').mean()
    
    ########################## ABSOLUTE ERRORS ########################## 
    pos = list(range(len(means.CC)))
    width=1./(len(methods) + 1)
    cmap = cm.rainbow(np.linspace(0,1,len(methods)))
    fig, ax =plt.subplots(figsize=(20,6))
    
    for i in xrange(len(methods)):
        plt.bar([p + width*i for p  in pos], means[methods[i]], width, color=cmap[i], label=methods[i])
    
    ax.set_ylabel('Absolute error')
    ax.set_title('Absolute errors for each class')
    ax.set_xticks([p + 2.5 * width for p in pos])
    ax.set_xticklabels(means.index)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30)
    
    plt.xlim(min(pos)-width, max(pos)+width*len(methods))
    plt.ylim([0, 
                #means.max().max()
                0.5
                ])
    
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    pp.savefig(fig)
    
    means_without_stephanopyxis = means.drop(['Stephanopyxis'], axis=0, inplace=False)
    
    pos = list(range(len(means_without_stephanopyxis.CC)))
    width=1./(len(methods) + 1)
    cmap = cm.rainbow(np.linspace(0,1,len(methods)))
    fig, ax =plt.subplots(figsize=(20,6))
    
    for i in xrange(len(methods)):
        plt.bar([p + width*i for p  in pos], means_without_stephanopyxis[methods[i]], width, color=cmap[i], label=methods[i])
    
    ax.set_ylabel('Absolute error')
    ax.set_title('Absolute errors for each class without Stephanopyxis')
    ax.set_xticks([p + 2.5 * width for p in pos])
    ax.set_xticklabels(means_without_stephanopyxis.index)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30)
    
    plt.xlim(min(pos)-width, max(pos)+width*len(methods))
    plt.ylim([0, 
                #means.max().max()
                0.5
                ])
    
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    pp.savefig(fig)
    #####################################################################
    
    ##################### RELATIVE ABSOLUTE ERRORS ###################### 
    
    pos = list(range(len(r_means.CC)))
    width=1./(len(methods) + 1)
    cmap = cm.rainbow(np.linspace(0,1,len(methods)))
    fig, ax =plt.subplots(figsize=(20,6))
    
    for i in xrange(len(methods)):
        plt.bar([p + width*i for p  in pos], r_means[methods[i]], width, color=cmap[i], label=methods[i])
    
    ax.set_ylabel('Relative absolute error')
    ax.set_title('Relative absolute errors for each class')
    ax.set_xticks([p + 2.5 * width for p in pos])
    ax.set_xticklabels(r_means.index)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30)
    
    plt.xlim(min(pos)-width, max(pos)+width*len(methods))
    plt.ylim([0, 
                r_means.max().max() * 1.2
                #0.5
                ])
    
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    pp.savefig(fig)
    
    rmeans_without_stephanopyxis = r_means.drop(['Stephanopyxis'], axis=0, inplace=False)
    pos = list(range(len(rmeans_without_stephanopyxis.CC)))
    width=1./(len(methods) + 1)
    cmap = cm.rainbow(np.linspace(0,1,len(methods)))
    fig, ax =plt.subplots(figsize=(20,6))
    
    for i in xrange(len(methods)):
        plt.bar([p + width*i for p  in pos], rmeans_without_stephanopyxis[methods[i]], width, color=cmap[i], label=methods[i])
    
    ax.set_ylabel('Relative absolute error without Stephanopyxis')
    ax.set_title('Relative absolute errors for each class')
    ax.set_xticks([p + 2.5 * width for p in pos])
    ax.set_xticklabels(rmeans_without_stephanopyxis.index)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30)
    
    plt.xlim(min(pos)-width, max(pos)+width*len(methods))
    plt.ylim([0, 
                r_means.max().max() * 1.2
                #0.5
                ])
    
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    pp.savefig(fig)
    #####################################################################
    
    
    
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    colors = [(r/255., g/255., b/255.) for (r,g,b) in colors]
    
    for cls, pred in preds.iteritems():
        best_method=r_means.loc[cls].argmin()
        fig, ax =plt.subplots(figsize=(16,9))
        plt.plot(pred.CC, color=colors[2], label='CC', lw=1.5)
        if best_method != 'CC':
            plt.plot(pred[best_method], color=colors[1], label=best_method, lw=1.5)
            
        best_method2=means.loc[cls].argmin()
        if best_method2 != 'CC' and best_method2 != best_method:
            plt.plot(pred[best_method2], color=colors[3], label=best_method2, lw=1.5)
        plt.plot(pred.True, color=colors[0], label='True', lw=1.5)
        plt.legend()
        plt.ylabel('Prevalence of positive class')
        plt.xlabel('Samples')
        plt.title("Quantification methods for {}.".format(cls, best_method))
        plt.xlim((0, len(pred)))
        if y_max == 'fixed':
            ymax = 1.
        else:
            ymax = max([pred.CC.max(), pred[best_method].max(), pred[best_method2].max(), pred.True.max()]) * 1.2
        plt.ylim((0, ymax))
        plt.grid()
        pp.savefig(fig)
        
    pp.close()
    
    folder, file = os.path.split(pdf_path)
    means.to_csv(folder + '/AbsoluteErrors.csv')
    r_means.to_csv(folder + '/RelativeAbsoluteErrors.csv')
    true_negatives.to_csv(folder + '/TrueNegatives.csv')
    
    
path = '/Users/administrador/Dropbox/experimentos/Plancton-WHOI/resultados/OneVsAll51ClassesAndMixDL/'
pdf_path = '/Users/administrador//Dropbox/ExperimentosPlacton/Cjto-WHOI/resultados/OneVsAll1vs50DL/'

get_results(path, pdf_path + 'figuresWithoutNorm.pdf', False, y_max='fixed')
get_results(path, pdf_path + 'figuresWithoutNormScaled.pdf', False, y_max='var')
get_results(path, pdf_path + 'figures.pdf', True, y_max='fixed')
get_results(path, pdf_path + 'figuresScaled.pdf', True, y_max='var')



