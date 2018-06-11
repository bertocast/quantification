import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV

from quantification.cc.base import BaseClassifyAndCountModel, PCC, PAC
from quantification.metrics import  binary_kl_divergence
from quantification.metrics.ordinal import emd
from quantification.ordinal.qtree import QTree


class OrderedQuantificationTreeFH(BaseClassifyAndCountModel):
    """
            Ordered Quantification Tree -> method one vs all the rest (binary classification)

            Parameters
            -----------
            estimator_class : object, optional
                An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
                the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

            estimator_params : dictionary, optional
                Additional params to initialize the classifier.

            estimator_grid : dictionary, optional
                During training phase, grid search is performed. This parameter should provided the parameters of the classifier
                that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

            ****parametros propios, si se necesitan, van aqui ---->>>>>>JAIMEEEEE
            usingPAC : bool
                si es false (defecto) se usa PCC como clasificador binario
                si se pone a true se usa PAC como clasificador binario

            Attributes
            ----------
            tree_ : QTree
                Arbol de modelos segun el metodo de Sebastiani

             oqn_estimators_ : array
                [estimador_binario, classes_left, classes_right, KLD]
                El estimador binario sera tipo PCC o PCA

            """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, method=None, usingPAC=False):
        super(OrderedQuantificationTreeFH,self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        # parametros especificos aqui
        self.usingPAC = usingPAC
        self.method = method
        self.oqn_estimators_ = None
        self.tree = None


    def __binarize_labels(self, y, classes_izda):
        """
        Adapta las etiquetas a las DOS clases que estoy enfrentando (poniendo mascaras)
        Convenio: Clases de la izquierda -> Pongo +1, Clases de la derecha-> Pongo -1

        :param y:              etiquetas originales
        :param classes_izda:   classes_izda forman la clase de la izquierda (+1)
        :return:               y_bin: etiquetas binarizadas
        """
        mask = np.in1d(y, classes_izda)  #in1d retorna true en las posiciones de y que tienen los valores de classes_izda
        y_bin = np.ones(y.shape, dtype=np.int)
        y_bin[~mask] = -1
        #y_bin[mask] = -1
        return y_bin


    def __build_recursive_tree(self, klds, classes_in_subtree, classes_covered):
        """
           klds --> array de klds y las posiciones de sus estimadores en qnd.estimators_ . Cada estimador es:
                    np.array[estimador_PCC, classes_left, classes_right, KLD]
           classes_in_subtree --> las clases que van por cada rama (subarbol)
           classes_covered --> clases ya cubiertas por niveles superiores del arbol
        """
        n_estimators = klds.shape[0]

        #Escojo el estimador de mejor KLD
        pos_best_kld= np.argmin(klds[:,0])
        pos_best_estimator= int(klds[pos_best_kld][1])
        estimator = self.oqn_estimators_[pos_best_estimator]

        # separo los valores kld que van a ir a los subarboles izquierdo y derecho
        klds_left = klds[:pos_best_kld]
        klds_right = klds[pos_best_kld+1:]

        # obtengo las clases a la izquierda y a la derecha del clasificador binario
        classes_left = np.array(estimator[1])  # ojo, esto son los nonmbres de las clases de la izquierda
        classes_right = np.array(estimator[2]) # ojo, esto son los nombres de las clases de la derecha

        # obtengo las clases que quedan por tratar en cada subarbol
        classes_left_subtree = np.setdiff1d(classes_in_subtree, classes_right)     # van en la rama izquierda
        classes_left_pend = np.setdiff1d(classes_left_subtree, classes_covered)   # dejo las no tratadas
        classes_right_subtree = np.setdiff1d(classes_in_subtree, classes_left)  # van en la rama derecha
        classes_right_pend = np.setdiff1d(classes_right_subtree, classes_covered) # dejo las no tratadas

        #CASO BASE
        if n_estimators == 1:  #llegamos al ultimo nivel
            class_left = classes_left_pend[0]
            class_right = classes_right_pend[0]
            classes_covered.append(class_left)
            classes_covered.append(class_right)
            left_tree = QTree(class_left)
            right_tree = QTree(class_right)

        #resto de casos: el n_estimadores por tratar es mayor que 1
        elif len(classes_left_pend) == 1:  # final por la rama izquierda
            the_class=classes_left_pend[0]
            left_tree = QTree(the_class)
            classes_covered.append(the_class)
            right_tree = self.__build_recursive_tree(deepcopy(klds_right), classes_right_pend.tolist(), classes_covered)

        elif len(classes_right_pend) == 1: # final por la rama derecha
            the_class=classes_right_pend[0]
            right_tree = QTree(the_class)
            classes_covered.append(the_class)
            left_tree = self.__build_recursive_tree(deepcopy(klds_left), classes_left_pend.tolist(), classes_covered)

        else:  # caso general
            left_tree = self.__build_recursive_tree(deepcopy(klds_left), classes_left_pend.tolist(), classes_covered)
            right_tree = self.__build_recursive_tree(deepcopy(klds_right), classes_right_pend.tolist(), classes_covered)

        label = str(classes_left.tolist())+" vs "+str(classes_right.tolist())
        return QTree(label, pos_best_estimator, left_tree , right_tree)


    def generate_tree(self, X_val, y_val, verbose=False):
        """
        Genera el arbol con los modelos que hay en self.oqn_estimators_ (tipo PCC)
        Para comparar modelos y construir el arbol se usan las distancias KLD (que se calculan aqui)
                oqn_estimators_ --> np.array[estimador_PCC, classes_left, classes_right, KLD]

        :param X_val: ejemplos para validacion
        :param y_val: etiquetas para validacion
        :param verbose:
        :return: arbol binario generado con los estimadores
        """

        # PASO 1: Obtengo las distancias KLD
        # multiclass: KLD necesita una lista las prevalencias de cada clase
        # binario: KLD necesita una prevalencia nada mas
        n_estimators=len(self.oqn_estimators_)
        klds = []
        for i in range(n_estimators):
            clf=self.oqn_estimators_[i][0]
            classes_left=self.oqn_estimators_[i][1]
            #classes_right = self.oqn_estimators_[i][2]

            #Uso siempre PCC
            #p_pred=clf._predict_pcc(X_val)    #version vieja de Alberto
            p_pred = (clf._predict_pcc(X_val))[1] # la version nueva de Alberto no tiene quant. binarios y predict retorna [p_false p_true]


            y_val_bin=self.__binarize_labels(y_val,classes_left)  #classes_left vs el resto
            # Get the true prevalence, i.e., percentage of positives samples in each of the classes.
            num_true = np.count_nonzero(y_val_bin==1)
            samples = len(y_val_bin)
            p_true = num_true / float(samples)  # la(s) clase(s) left vale +1, el resto -1

            kld_score = binary_kl_divergence(p_true, p_pred)
            klds.append([kld_score,i])
            self.oqn_estimators_[i][3]=kld_score

            if (verbose):
                print( "KLD del modelo ",i,":")
                print("--> p_pred =", p_pred, 1-p_pred)
                print("--> p_true =", p_true, 1-p_true, samples-num_true)
                print("--> kld_score =", kld_score)

        # PASO 2: Genero el arbol a partir de las distancias KLD (que van dentro de los estimadores)
        # uso indices para ahorrar memoria
        self.tree= self.__build_recursive_tree(np.asarray(klds),self.classes_, classes_covered=[])
        return self


    def fit(self, X, y, cv=50, verbose=False):
        """
            Crea todos los estimadores PCC/PAC de unas clases frente a otras y los guarda en un array numpy
            self.oqn_estimartors_ con este formato:
                oqn_estimators_ --> [estimador_PCC/PAC, classes_left, classes_right, KLD]
        """

        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)

        l_estimators = []

        #se recorren todas la clases menos la ultima
        for pos_class in range(n_classes-1):
            if verbose:
                print ("Fitting classifier for class [1..{}] vs [{}..{}]".format(pos_class+1, pos_class+2, n_classes))

            #clases 0..pos_class frente al resto
            classes_left= self.classes_[:pos_class + 1]
            classes_right= self.classes_[pos_class + 1:]

            y_bin = self.__binarize_labels(y, classes_left)  #classes_left vs classes_right

            #se invoca un clasificador de quantificacion binario PCC/PAC
            if self.usingPAC:
                clf = PAC(self.estimator_class, self.estimator_params, self.estimator_grid, self.grid_params)
            else:
                clf = PCC(self.estimator_class, self.estimator_params, self.estimator_grid, self.grid_params)

            clf.fit(X, y_bin, cv, verbose)

            # se guardan los modelos y las clases con las que se obtienen vs las otras (el 0 es para luego->KLD)
            l_estimators.append([deepcopy(clf), classes_left, classes_right, 0])

            if verbose==True:
                num_true = np.count_nonzero(y_bin==1)
                samples = len(y_bin)
                p_true = num_true / float(samples)  # la(s) clase(s) left vale +1, el resto -1
                p_prev = clf.predict(X) #esto es reescritura
                print("prevalencia =", p_prev, "prob_true =", p_true, "num_true =", num_true )
                if self.usingPAC==True:
                    print("prevalencia_pcc (para comprobar la correcion) =", (clf._predict_pcc(X))[1], "prob_true =", p_true, "num_true =", num_true)


        #convierto la lista de estimadores en un numpy array
        self.oqn_estimators_ = np.array(l_estimators)
        self.generate_tree(X, y, verbose)
        return self



    def predict(self, X, method_=None):
        """ Predict using one of the available methods
            Retorna la prevalencia de cada clase, basado las probabilidades calculadas a partir de los modelos
            Rerorna la prevalencia de cada clase contando las clases que tienen mas probabilidad, es decir,
                calculando la frecuencia relativa de clada clase (contando las hojas donde cae cada ejemplo)

            Parameters
            ---------
            X : numpy array, shape = (n_samples, n_features)
                Samples.

            method_ : string, optional, default 'None'
                Method to use in the prediction. It can be one of:
                    - 'sebas' : Sebastiani Method Tree Based for Ordinal Quantification
                    - 'sebas_hojas' : Sebastiani Method Tree Based for Ordinal Quantification contando hojas
                    - 'sebas_freq' : Sebastiani Method Tree Based for Ordinal Quantification contando hojas
                    - 'pcc_ord' : Probabilistic Classify & Count predictions for Ordinal Quantification (using probs samples from models)
                    - 'pcc_ord_freq' : Probabilistic Classify & Count predictions for Ordinal Quantification (using probs samples from models and computign frequencies)
                    - 'pcc_ord_nosamples' : Probabilistic Classify & Count predictions for Ordinal Quantification (using probs classes from models)
                    - 'pac_ord_nosamples' : Probabilistic Adjusted Count predictions for Ordinal Quantification (only version probs classes from models)

            Returns
            -------
            pred : numpy array, shape = (n_classes)
                Prevalences of each of the classes. Note that  sum(pred[i]) = 1
        """

        if method_ == None:
            if self.method == None:
                raise ValueError("Invalid method %s.", method_)
            else:
                 method_ = self.method

        if method_ == 'sebas':
            return self._predict_sebas(X)
        elif method_ == 'sebas_freq':
            return self._predict_sebas_freq(X)
        elif method_ == 'sebas_hojas':
            return self._predict_sebas_hojas(X)
        elif method_ == 'pcc_ord':
            return self._predict_pcc_ord(X)
        elif method_ == 'pcc_ord_freq':
            return self._predict_pcc_ord_freq(X)
        elif method_ == 'pcc_ord_nosamples':
            return self._predict_pcc_ord_nosamples(X)
        elif method_ == 'pac_ord_nosamples':
            return self._predict_pac_ord_nosamples(X)
        else:
            raise ValueError("Invalid method %s.", method_)



    #*********************************************************************************************************
    #   Predicciones con versiones del arbol de Sebastiani
    # *********************************************************************************************************

    def __compute_post_prob(self, node, X, probs_samples, classes_in_subtree, classes_covered):
        #Recibe los ejemplos y las probabilidades de cada ejemplo para cada clase (a 1 inicialmente)

        #caso base
        if node._is_leaf():
            classes_covered.append(node.label)
        else: #not node._is_leaf()
            oqn_estimator = self.oqn_estimators_[node.pos_model]
            current_estimator =  oqn_estimator[0] # Es un PCC
            classes_l= oqn_estimator[1]   #BORRAR
            classes_r  = oqn_estimator[2] #BORRAR
            classes_left = np.array(oqn_estimator[1])
            classes_right = np.array(oqn_estimator[2])

            #OJO!!!!!!
            #_predictions da las predicciones sin corregir (modo PCC) incluso si los estimadores son PAC
            # predict si que devuelve cosas distintas en PCC y en PAC

            #prediciones para cada ejemplo del estimador binario PCC para las clases de la izda vs las de de la derecha
            #predictions[0] es la probabilidad de la clase -1 (derecha) y predictions[1] de la clase 1 (izquierda)
            #predictions = current_estimator._predictions(X) #version anterior -> mejor no meter nuevos metodos en los cuantificadores
            predictions = current_estimator.estimators_[1].predict_proba(X) #version nueva de Alberto


            p = np.mean(predictions, axis=0) #p[0]= prevalencia de la clase -1, p[1]= prevalencia de la clase +1
            predictions_left = predictions[:, 1]
            predictions_right = predictions[:, 0]

            # en la izquierda van las clases que en clasficacion binaria son la clase +1
            #hay que tener cuidado con las clases que ya se han considerado antes en el arbol!!!!!
            #y con las que no son de la rama correspondiente!!!!!!
            if len(classes_in_subtree)==0:  #primera llamada recursiva
                classes_left_subtree = classes_left
                classes_right_subtree = classes_right
            else:
                classes_left_subtree = np.setdiff1d(classes_in_subtree, classes_right)
                classes_right_subtree = np.setdiff1d(classes_in_subtree, classes_left)

            for cl in classes_left_subtree:
                pos_cl=self.classes_.index(cl)
                if not cl in classes_covered: #multiplico por su probabilidad
                    probs_samples[:, pos_cl] *= predictions_left
            # en la derecha van las clases que en clasficacion binaria son la clase -1
            for cl in classes_right_subtree:
                pos_cl = self.classes_.index(cl)
                if not cl in classes_covered:
                    probs_samples[:, pos_cl] *= predictions_right   #ojo!!!

            self.__compute_post_prob(node.left, X, probs_samples, classes_left_subtree, classes_covered)
            self.__compute_post_prob(node.right, X, probs_samples, classes_right_subtree, classes_covered)


    def _predict_sebas(self, X):
        """"
            Se predice usando el arbol de Sebastiani
            Para cada ejemplo, baja por los nodos del arbol hasta la hoja que contenga la clase correspondiente
            Hay que ir "aumulando" las predicciones de los estimadores de cada nodo interno recorrido
            Si type=='probs_samples' se devuelven las predicciones para cada ejemplo de X
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        if self.tree==None:
            raise ValueError('No se ha generado el arbol interno de Sebastiani')
        probs_samples=np.ones((n_samples, n_classes))
        #modifica las probabiliades de las clases bajando por el arbol
        self. __compute_post_prob(self.tree, X, probs_samples, [], [])
        return np.mean(probs_samples, axis=0)


    def _predict_sebas_freq(self, X):
        """"
            Se predice usando el arbol de Sebastiani
            Para cada ejemplo, baja por los nodos del arbol hasta la hoja que contenga la clase correspondiente
            Hay que ir "aumulando" las predicciones de los estimadores de cada nodo interno recorrido
            Si type=='probs_samples' se devuelven las predicciones para cada ejemplo de X
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        if self.tree==None:
            raise ValueError('No se ha generado el arbol interno de Sebastiani')
        probs_samples=np.ones((n_samples, n_classes))
        #modifica las probabiliades de las clases bajando por el arbol
        self. __compute_post_prob(self.tree, X, probs_samples, [], [])

        classes_samples = np.argmax(probs_samples, axis=1)
        unique_elements, counts_elements = np.unique(classes_samples, return_counts=True)
        # Por si no todas las clases estan representadas
        cont = np.zeros(len(self.classes_))
        cont[unique_elements] = counts_elements
        return cont / float(np.sum(cont))  # frecuencias relativas


    def __obtain_leaves(self, node, X, cont_samples_classes):
        #Recibe los ejemplos y cuenta las ramas (clases) en las que cae cada ejemplo

        #caso base
        if node._is_leaf():
            pass
        else: #not node._is_leaf()
            oqn_estimator = self.oqn_estimators_[node.pos_model]
            current_estimator =  oqn_estimator[0] # Es un PCC
            classes_l= oqn_estimator[1]   #BORRAR
            classes_r  = oqn_estimator[2] #BORRAR
            classes_left = np.array(oqn_estimator[1])
            classes_right = np.array(oqn_estimator[2])

            #prediciones para cada ejemplo del estimador binario PCC para las clases de la izda vs las de de la derecha
            #predictions[0] es la probabilidad de la clase -1 (derecha) y predictions[1] de la clase 1 (izquierda)
            #predictions = current_estimator._predictions(X) #version anterior -> mejor no meter nuevos metodos en los cuantificadores
            predictions = current_estimator.estimators_[1].predict_proba(X) #version nueva de Alberto

            predictions_left = predictions[:, 1]
            predictions_right = predictions[:, 0]

            samples_left = np.flatnonzero(predictions_left>= 0.5)
            samples_right = np.flatnonzero(predictions_left< 0.5)

            for cl in classes_left:
                pos_cl=self.classes_.index(cl)
                cont_samples_classes[samples_left, pos_cl]+=1

            for cl in classes_right:
                    pos_cl = self.classes_.index(cl)
                    cont_samples_classes[samples_right, pos_cl] += 1

            self.__obtain_leaves(node.left, X, cont_samples_classes)
            self.__obtain_leaves(node.right, X, cont_samples_classes)


    def _predict_sebas_hojas(self, X):
        """"
            Se predice usando el arbol de Sebastiani
            Para cada ejemplo, baja por los nodos del arbol hasta la hoja que contenga la clase correspondiente
            Se baja por la rama para la que el ejemplo tenga mas probabilidad
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        if self.tree==None:
            raise ValueError('No se ha generado el arbol interno de Sebastiani')
        cont_samples_classes=np.zeros((n_samples, n_classes))
        #modifica las probabiliades de las clases bajando por el arbol
        self. __obtain_leaves(self.tree, X, cont_samples_classes)

        classes_samples = np.argmax(cont_samples_classes, axis=1)
        unique_elements, counts_elements = np.unique(classes_samples, return_counts=True)
        # Por si no todas las clases estan representadas
        cont = np.zeros(len(self.classes_))
        cont[unique_elements] = counts_elements
        return cont / float(np.sum(cont))  # frecuencias relativas


    def _predictions_sebas(self, X):
        """
            Returns
            -------
                probs_samples : numpy array, shape = (n_samples, n_classes)
                    Prevalences of each of the classes for each sample of X. Note that  sum(predictions[i][j]) = 1
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        if self.tree==None:
            raise ValueError('No se ha generado el arbol interno de Sebastiani')
        probs_samples=np.ones((n_samples, n_classes))
        #modifica las probabiliades de las clases bajando por el arbol
        self. __compute_post_prob(self.tree, X, probs_samples, [], [])
        return probs_samples


    # *********************************************************************************************************
    #   Predicciones con versiones del arbol usando PCC y "metodo de las restas"
    # *********************************************************************************************************

    def __correct_probabilities(self, probs_sample):
        """
        Retorna las probabilidades de un ejemplo para que sean crecientes
        :param probs_sample: probabilidades de los modelos para un ejemplo
        :return: probabilidades corregidas -> monotonas (crecientes la parte izda)
        """
        n_models = len(probs_sample)
        probs1 = np.copy(probs_sample)
        for i in range(1,n_models):  # de izda a dcha
            if probs1[i] < probs1[i-1]:
                probs1[i] = probs1[i-1]
        probs2 = np.copy(probs_sample)
        for i in range(n_models-1,0,-1):  # de dcha a izda
            if probs2[i] < probs2[i-1]:
                probs2[i-1] = probs2[i]
        return (probs1+probs2)/2


    def __check_probabilities(self, predictions_bin):
        """
            Uso un array donde el numero de filases igual al numero de ejmplos
            Cada fila es otro array con las predicciones de los estimadores binarios para ese ejemplo
            Chequea si las prediciones de cada estimador son crecientes (en la parte izquierda)
            Si no lo son: corregir de izda-dcha y de decha-izquiera y quedarse con la media
        """
        #si no son crecientes se corrige
        crecientes = np.all(predictions_bin[:,1:] >= predictions_bin[:,:-1],axis=1)
        samples_to_correct = np.nonzero(crecientes != True)  #ojo, retorna una tupla
        for sample in samples_to_correct[0]:
                pnew = self.__correct_probabilities(predictions_bin[sample])
                predictions_bin[sample] = pnew


    def _predict_pcc_ord(self, X):
        """"
            Metodo basado en el articulo de  Frank and Hall "modificado"!!!!!. Con PCC para los clasificadores binarios
            En lugar de la formula de FH original, se usa esta otra para CADA EJEMPLO:
                p_v1= probs_left - suma(probs_clases_anteriores)    -> metodo de las restas

            Puede ser necesario corregir las probabilidades de algunos ejemplos para que sean --> monotonas.
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # ******* CAMBIO DE ESTA VERSION
        # Asegurarse de que las probabilidades de los modelos son "monotonas"
        # (crecientes si miramos la partes izquierda o decrecientes para la parte derecha)

        # uso un array donde el numero de filas es igual al numero de ejemplos
        # cada fila es otro array con las predicciones de cada estimador binario para ese ejemplo
        predictions_bin = np.zeros((n_samples, n_classes-1))
        #predict_bin = np.ndarray(n_classes-1)
        for i in range(n_classes-1):  # hay un estimador binario menos que clases
            clf = self.oqn_estimators_[i][0]
            #p = clf._predictions(X) #version anterior -> mejor no meter nuevos metodos en los cuantificadores
            p = clf.estimators_[1].predict_proba(X) #version nueva de Alberto

            predictions_bin[:,i] = p[:,1]  #solo meto las prob true de la parte izda del modelo binario

        self.__check_probabilities(predictions_bin)

        # ahora calculo para cada ejemplo las probabiliades de ser de cada clase
        probs_classes = np.zeros((n_samples,n_classes))     #para guardar las probabilidades de cada clase para cada ejemplo
        # recorro los modelos en orden de la descomposicion FH  (de izda a derecha)
        # "vi" vale tanto para la clase que trato como para la posicion del modelo en el que estoy
        # modelo: {vo,v1,..,vi}vs{vi+1,...,vk}  con k=n_classes-1  {left}vs{right}
        # p_vo y p_vk ya se saben porque son los extremos de un calificador binario
        # para las otras clases calculo la "diferencias de probabilidades debidas a la clase que se considera"
        probs_primer_modelo_left = predictions_bin[:,0]
        probs_ultimo_modelo_left = predictions_bin[:,n_classes-2]
        probs_classes[:,0] = probs_primer_modelo_left                # p_vo
        probs_classes[:,n_classes-1] = 1 - probs_ultimo_modelo_left  # p_vk
        for vi in range(1, n_classes - 1):       # recordar que la clase "vi" es la ultima de la parte izquierda
            probs_left = predictions_bin[:,vi]     # probs_left: parte izda del modelo actual
            dif =  probs_left - np.sum(probs_classes[:,0:vi],axis=1)
            if min(dif)<-0.0001:
                print (probs_left)
                print (probs_classes)
                raise ValueError("Diferencia entre clases negativas in _predict_pcc_ord")
            probs_classes[:,vi] = abs(dif)

        return np.mean(probs_classes, axis=0)



    def _predict_pcc_ord_freq(self, X):
        """"
            Igual que el metodo anterior pero contado las clases que tienen mas probabilidad (deberia ir peor)
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # ******* CAMBIO DE ESTA VERSION
        # Asegurarse de que las probabilidades de los modelos son "monotonas"
        # (crecientes si miramos la partes izquierda o decrecientes para la parte derecha)

        # uso un array donde el numero de filas es igual al numero de ejemplos
        # cada fila es otro array con las predicciones de cada estimador binario para ese ejemplo
        predictions_bin = np.zeros((n_samples, n_classes - 1))
        # predict_bin = np.ndarray(n_classes-1)
        for i in range(n_classes - 1):  # hay un estimador binario menos que clases
            clf = self.oqn_estimators_[i][0]
            #p = clf._predictions(X) #version anterior -> mejor no meter nuevos metodos en los cuantificadores
            p = clf.estimators_[1].predict_proba(X) #version nueva de Alberto
            predictions_bin[:, i] = p[:, 1]  # solo meto las prob true de la parte izda del modelo binario

        self.__check_probabilities(predictions_bin)

        # ahora calculo para cada ejemplo las probabiliades de ser de cada clase
        probs_classes = np.zeros((n_samples, n_classes))  # para guardar las probabilidades de cada clase para cada ejemplo
        # recorro los modelos en orden de la descomposicion FH  (de izda a derecha)
        # "vi" vale tanto para la clase que trato como para la posicion del modelo en el que estoy
        # modelo: {vo,v1,..,vi}vs{vi+1,...,vk}  con k=n_classes-1  {left}vs{right}
        # p_vo y p_vk ya se saben porque son los extremos de un calificador binario
        # para las otras clases calculo la "diferencias de probabilidades debidas a la clase que se considera"
        probs_primer_modelo_left = predictions_bin[:, 0]
        probs_ultimo_modelo_left = predictions_bin[:, n_classes - 2]
        probs_classes[:, 0] = probs_primer_modelo_left  # p_vo
        probs_classes[:, n_classes - 1] = 1 - probs_ultimo_modelo_left  # p_vk
        for vi in range(1, n_classes - 1):  # recordar que la clase "vi" es la ultima de la parte izquierda
            probs_left = predictions_bin[:, vi]  # probs_left: parte izda del modelo actual
            dif = probs_left - np.sum(probs_classes[:, 0:vi], axis=1)
            if min(dif) < -0.0001:
                print (probs_left)
                print (probs_classes)
                raise ValueError("Diferencia entre clases negativas in _predict_pcc_ord")
            probs_classes[:, vi] = abs(dif)

        classes_samples = np.argmax(probs_classes, axis=1)
        unique_elements, counts_elements = np.unique(classes_samples, return_counts=True)
        # Por si no todas las clases estan representadas
        cont = np.zeros(len(self.classes_))
        cont[unique_elements] = counts_elements
        return cont / float(np.sum(cont))  # frecuencias relativas


    def _predict_pcc_ord_nosamples(self, X):
        """"
            Metodo basado en el articulo de Frank and Hall "modificado"!!!!!.
            Con PCC para los clasificadores binarios
            En lugar de la formula de FH original, se usa esta otra a NIVEL DE MODELO (no de ejemplo):
                p_v1= probs_left - suma(probs_clases_anteriores)    -> metodo de las restas
        """
        n_classes = len(self.classes_)

        # uso un array donde el numero de filas es igual al numero clases
        predict_bin = np.zeros(n_classes-1)
        for i in range(n_classes-1):  # hay un estimador binario menos que clases
            clf = self.oqn_estimators_[i][0]
            # ojo: predict si que devuelve cosas distintas en PCC y en PAC
            #p = clf._predict_pcc(X)  # version vieja de Alberto (al se un estimador binario da la prevalencia true)
            p = (clf._predict_pcc(X))[1] # la version nueva de Alberto no tiene quant. binarios y predict retorna [p_false p_true]

            predict_bin[i] = p  #solo meto las prob true de la parte izda del modelo binario

        #si no son crecentes se corrige
        if not np.all(predict_bin[1:] >= predict_bin[:-1]):
            predict_bin=self.__correct_probabilities(predict_bin)

        # recorro los modelos en orden de la descomposicion FH  (de izda a derecha)
        # "vi" vale tanto para la clase que trato como para la posicion del modelo en el que estoy
        # modelo: {vo,v1,..,vi}vs{vi+1,...,vk}  con k=n_classes-1  {left}vs{right}
        # p_vo y p_vk ya se saben porque son los extremos de un calificador binario
        # para las otras clases calculo la "diferencias de probabilidades debidas a la clase que se considera"
        probs_classes = np.zeros(n_classes)
        for vi in range( n_classes - 1):       # recordar que la clase "vi" es la ultima de la parte izquierda
            p_left = predict_bin[vi]
            p_vi = p_left - sum(probs_classes[0:vi])
            if (p_vi)<-0.0001:
                raise ValueError("Diferencia entre clases negativas in _predict_pcc_ord_nosamples")
            probs_classes[vi]= p_vi

        #falta poner la probabilidad de la ultima clase
        probs_classes[n_classes-1] = 1 - sum(probs_classes[0:n_classes-1])

        return probs_classes


    def _predict_pac_ord_nosamples(self, X):
        """"
            Metodo basado en el articulo de Frank and Hall "modificado"!!!!!.
            Con PAC para los clasificadores binarios  -> probabilidades corregidas
            En lugar de la formula de FH original, se usa esta otra a NIVEL DE MODELO (no de ejemplo):
                p_v1= probs_left - suma(probs_clases_anteriores)    -> metodo de las restas
        """

        n_classes = len(self.classes_)

        # uso un array donde el numero de filas es igual al numero clases
        predict_bin = np.zeros(n_classes-1)
        for i in range(n_classes-1):  # hay un estimador binario menos que clases
            clf = self.oqn_estimators_[i][0]
            # ojo: predict si que devuelve cosas distintas en PCC y en PAC
            #p = clf._predict_pac(X)  # version vieja de Alberto (al se un estimador binario da la prevalencia true)
            p= (clf._predict_pac(X))[1] # la version nueva de Alberto no tiene quant. binarios y predict retorna [p_false p_true]
            #print(i,p)

            predict_bin[i] = p  #solo meto las prob true de la parte izda del modelo binario

        #si no son crecentes se corrige
        if not np.all(predict_bin[1:] >= predict_bin[:-1]):
            predict_bin=self.__correct_probabilities(predict_bin)

        # recorro los modelos en orden de la descomposicion FH  (de izda a derecha)
        # "vi" vale tanto para la clase que trato como para la posicion del modelo en el que estoy
        # modelo: {vo,v1,..,vi}vs{vi+1,...,vk}  con k=n_classes-1  {left}vs{right}
        # p_vo y p_vk ya se saben porque son los extremos de un calificador binario
        # para las otras clases calculo la "diferencias de probabilidades debidas a la clase que se considera"
        probs_classes = np.zeros(n_classes)
        for vi in range(n_classes - 1):  # recordar que la clase "vi" es la ultima de la parte izquierda
            p_left = predict_bin[vi]
            p_vi = p_left - sum(probs_classes[0:vi])
            if (p_vi) < -0.0001:
                raise ValueError("Diferencia entre clases negativas in _predict_pac_ord_nosamples")
            probs_classes[vi] = p_vi

        #falta poner la probabilidad de la ultima clase
        probs_classes[n_classes-1] = 1 - sum(probs_classes[0:n_classes-1])

        return probs_classes


    def _predictions_pcc_ord(self, X):
        """
            Returns
            -------
                probs_classes : numpy array, shape = (n_samples, n_classes)
                    Prevalences of each of the classes for each sample of X. Note that  sum(predictions[i][j]) = 1
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # ******* CAMBIO DE ESTA VERSION
        # Asegurarse de que las probabilidades de los modelos son "monotonas"
        # (crecientes si miramos la partes izquierda o decrecientes para la parte derecha)

        # uso un array donde el numero de filas es igual al numero de ejemplos
        # cada fila es otro array con las predicciones de cada estimador binario para ese ejemplo
        predictions_bin = np.zeros((n_samples, n_classes-1))
        #predict_bin = np.ndarray(n_classes-1)
        for i in range(n_classes-1):  # hay un estimador binario menos que clases
            clf = self.oqn_estimators_[i][0]
            #p = clf._predictions(X) #version anterior -> mejor no meter nuevos metodos en los cuantificadores
            p = clf.estimators_[1].predict_proba(X) #version nueva de Alberto

            predictions_bin[:,i] = p[:,1]  #solo meto las prob true de la parte izda del modelo binario

        self.__check_probabilities(predictions_bin)

        # ahora calculo para cada ejemplo las probabiliades de ser de cada clase
        probs_classes = np.zeros((n_samples,n_classes))     #para guardar las probabilidades de cada clase para cada ejemplo
        # recorro los modelos en orden de la descomposicion FH  (de izda a derecha)
        # "vi" vale tanto para la clase que trato como para la posicion del modelo en el que estoy
        # modelo: {vo,v1,..,vi}vs{vi+1,...,vk}  con k=n_classes-1  {left}vs{right}
        # p_vo y p_vk ya se saben porque son los extremos de un calificador binario
        # para las otras clases calculo la "diferencias de probabilidades debidas a la clase que se considera"
        probs_primer_modelo_left = predictions_bin[:,0]
        probs_ultimo_modelo_left = predictions_bin[:,n_classes-2]
        probs_classes[:,0] = probs_primer_modelo_left                # p_vo
        probs_classes[:,n_classes-1] = 1 - probs_ultimo_modelo_left  # p_vk
        for vi in range(1, n_classes - 1):       # recordar que la clase "vi" es la ultima de la parte izquierda
            probs_left = predictions_bin[:,vi]     # probs_left: parte izda del modelo actual
            dif =  probs_left - np.sum(probs_classes[:,0:vi],axis=1)
            if min(dif)<-0.0001:
                print (probs_left)
                print (probs_classes)
                raise ValueError("Diferencia entre clases negativas in _predict_pcc_ord")
            probs_classes[:,vi] = abs(dif)

        return probs_classes



#******************************************************************************************************
# ****************************  WRAPPERS ************************************************
# ******************************************************************************************************

class OrderedQuantificationTreeFH_sebas(OrderedQuantificationTreeFH):
    def predict(self, X, method='sebas', plot=False):
        assert method == 'sebas'
        return self._predict_sebas(X)

class OrderedQuantificationTreeFH_sebas(OrderedQuantificationTreeFH):
    def predict(self, X, method='sebas_freq', plot=False):
        assert method == 'sebas_freq'
        return self._predict_sebas_freq(X)

class OrderedQuantificationTreeFH_hojas(OrderedQuantificationTreeFH):
    def predict(self, X, method='sebas_hojas', plot=False):
        assert method == 'sebas_hojas'
        return self._predict_sebas_freq(X)

class OrderedQuantificationTreeFH_pcc(OrderedQuantificationTreeFH):
    def predict(self, X, method='pcc_ord', plot=False):
        assert method == 'pcc_ord'
        return self._predict_pcc_ord(X)

class OrderedQuantificationTreeFH_pcc_freq(OrderedQuantificationTreeFH):
    def predict(self, X, method='pcc_ord_freq', plot=False):
        assert method == 'pcc_ord_freq'
        return self._predict_pcc_ord_freq(X)

class OrderedQuantificationTreeFH_pcc_nosamples(OrderedQuantificationTreeFH):
    def predict(self, X, method='pcc_ord_nosamples', plot=False):
        assert method == 'pcc_ord_nosamples'
        return self._predict_pcc_ord_nosamples(X)

class OrderedQuantificationTreeFH_pac(OrderedQuantificationTreeFH):
    def predict(self, X, method='pac_ord_nosamples', plot=False):
        assert method == 'pac_ord_nosamples'
        return self._predict_pac_ord_nosamples(X)



#******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************

class OrderedQuantificationTreeOVO(BaseClassifyAndCountModel):
    """
            Ordered Quantification Tree -> method one versus one(binary classification)

            Parameters
            -----------
            estimator_class : object, optional
                An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
                the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

            estimator_params : dictionary, optional
                Additional params to initialize the classifier.

            estimator_grid : dictionary, optional
                During training phase, grid search is performed. This parameter should provided the parameters of the classifier
                that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

            ****parametros propios, si se necesitan, van aqui ---->>>>>>JAIMEEEEE


            Attributes
            ----------
            oqn_estimators_ : array
                [estimador_binario, classes_left, classes_right]
                El estimador binario sera tipo PCC o PCA

            """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, method=None, usingPAC=False):
        super(OrderedQuantificationTreeOVO,self).__init__(estimator_class, estimator_params, estimator_grid, grid_params, b=None)
        # parametros especificos aqui
        self.usingPAC = usingPAC
        self.method = method
        self.oqn_estimators_ = None
        self.tree = None


    def __binarize_datasets(self, X, y, class_izda, class_dcha):
        """
        Adapta los conjuntos a las DOS clases que estoy enfrentando (poniendo mascaras)
        Convenio: Clase de la izquierda -> Pongo +1, Clase de la derecha-> Pongo -1
        :return:     X_bin, y_bin: conjuntos binarizadas
        """

        mask_izda = np.in1d(y, class_izda)  #in1d retorna true en las posiciones de y que tienen el valor de class_izda
        mask_dcha = np.in1d(y, class_dcha)  #in1d retorna true en las posiciones de y que tienen el valor de class_dcha
        mask= np.logical_or(mask_izda,mask_dcha)
        y_bin = np.ones(y.shape, dtype=np.int)
        y_bin[~mask_izda] = -1  #dejamos solo la parte izquierda con +1

        X_bin=X[mask]
        y_bin=y_bin[mask]

        #indices = np.flatnonzero(df.values == topic)
        #X_test_topic = X_test[indices]
        return X_bin, y_bin


    def fit(self, X, y, cv=50, verbose=False):
        """
            Crea todos los estimadores binarios PCC/PAC de una clases frente a otra y los guarda en un array numpy
            self.oqn_estimartors_ con este formato:
                oqn_estimators_ --> [estimador_PCC/PAC, class_left, class_right]
        """
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)

        l_estimators = []

        #se generan n*(n-1)/2 modelos
        for pos_class_i in range(n_classes-1):
            for pos_class_j in range (pos_class_i+1,n_classes):

                if verbose:
                    print ("Fitting classifier for class [{}] vs [{}]".format(pos_class_i+1, pos_class_j+1))

                class_left= self.classes_[pos_class_i]
                class_right= self.classes_[pos_class_j]

                X_bin, y_bin = self.__binarize_datasets(X, y, class_left, class_right)

                #se invoca un clasificador de quantificacion binario PCC/PAC
                if self.usingPAC:
                    clf = PAC(self.estimator_class, self.estimator_params, self.estimator_grid, self.grid_params)
                else:
                    clf = PCC(self.estimator_class, self.estimator_params, self.estimator_grid, self.grid_params)

                clf.fit(X_bin, y_bin, cv, verbose)

                #print("El grid usado:")  #BORRRARRRRRRRR
                #print(clf.estimator_grid)

                # se guardan los modelos y las clases con las que se obtienen vs las otras (el 0 es para luego->KLD)
                l_estimators.append([deepcopy(clf), class_left, class_right, 0])

                #BORRARR!!! es para depuracion
                if verbose==True:
                    num_true = np.count_nonzero(y_bin==1)
                    samples = len(y_bin)
                    p_true = num_true / float(samples)  # la(s) clase(s) left vale +1, el resto -1
                    p_prev = clf.predict(X_bin) #esto es reescritura
                    print("prevalencia =", p_prev, "prob_true =", p_true, "num_true =", num_true )

        #convierto la lista de estimadores en un numpy array
        self.oqn_estimators_ = np.array(l_estimators)
        return self


#****************************************************************************************************
# Predicciones con distintos metodos usando los modelos One versus One
# ****************************************************************************************************

    def predict(self, X, method_=None):
        """
            :param method_ : string, optional, default 'None'
                    - 'prob' :  retorna la prevalencia de cada clase,
                                basado las probabilidades calculadas a partir de los modelos
                    - 'freq' :  retorna la prevalencia de cada clase contando las clases que tienen mas probabilidad,
                                es decir, calculando la frecuencia relativa de clada clase
                                (contando las hojas donde cae cada ejemplo)
                    - 'hojas' : cada ejemplo es de la hoja a donde llega. Se cuenta cuantos hay de cada clase
                    - 'hojas_prob' : se llega hasta el nodo anterior a las hojas. Este nodo separa dos clases:
                                     para cada ejemplo, se asigna la probabiliad de las dos clases implicadas
        """

        if method_ == None:
            if self.method == None:
                raise ValueError("Invalid method %s.", method_)
            else:
                 method_ = self.method

        if method_ == 'prob':
            return self._predict_ovo(X)
        elif method_ == 'freq':
            return self._predict_ovo_freq(X)
        elif method_ == 'hojas':
            return self._predict_ovo_hojas(X)
        elif method_ == 'hojas_prob':
            return self._predict_ovo_hojas_prob(X)
        else:
            raise ValueError("Invalid method %s.", method_)



    def __traverse_tree_ovo(self, X, probs_samples, probs_samples_acum, pos_class_left, pos_class_right, level):
        """
        Se calcula la prob de cada ejemplo de X bajando por el arbol. Se multiplican las probs al bajar (solo las clases del subarbol)
        Hago una copia de las probabilidades que llevo acumuladas en cada llamada (coste espacil ineficiente pero no se me ocurre otra cosa)
        Al final, para las clases que siguen varios caminos hay que sunmar las probabiliades
        :param X: conjunto de ejemplos para clasificar con el arbol
        :param:probs_samples: probabilidades finales de cada clase para cada ejejmplo
        :param probs_samples_acum: probabiliades de cada clase para cada ejemplo acumuladas al bajar por cada rama
        :param pos_class_left: clase inicial de la rama (posicion, no el nombre de la clase) Inicial=0
        :param pos_class_right: clase final de la rama (posicion,no el nombre de la clase) Final=n_classes-1
        """
        n_classes = len(self.classes_)
        n_samples= X.shape[0]
        n_estim = (n_classes * n_classes - 1) / 2
        class_left = self.classes_[pos_class_left]
        class_right = self.classes_[pos_class_right]

        #caso base
        if (pos_class_left-pos_class_right==0):
            #print ("hoja", pos_class_left)
            probs_samples+=probs_samples_acum
        else:
            #obtengo el estimador de las dos clases consideradas
            i=0
            while i<n_estim and not (class_left==self.oqn_estimators_[i][1] and  class_right==self.oqn_estimators_[i][2]):
                i+=1
            estimador=self.oqn_estimators_[i][0]

            # ojo: pred son pares de la forma [prob_fallar prob_acertar]  -> prob_left=prob_acertar
            # obtengo las predicciones para cada ejemplo
            #predictions = estimador._predictions(X)  #version anterior -> mejor no meter nuevos metodos en los cuantificadores

            # En la nueva version de Alberto ya no hay quatificadores binarios especificos.
            # Son un caso particular de los multiclase, que tienen un diccionario de clasificadores binarios (one vs all)
            # Yo me quedo con el primero del diccionario porque se que es binario y solo hay uno
            predictions = estimador.estimators_[1].predict_proba(X)

            #samples_left = np.flatnonzero(predictions[:, 0] <= 0.5) indices de los ejemplos de la izquierda
            #samples_right = np.flatnonzero(predictions[:, 0] > 0.5)
            predictions_left = predictions[:, 1]
            predictions_right = predictions[:, 0]

            probs_samples_prev_left = np.zeros((n_samples,n_classes))
            probs_samples_prev_right = np.zeros((n_samples,n_classes))

            for cl in range(pos_class_left,pos_class_right): #parte inzquierda
                probs_samples_prev_left[:, cl] = probs_samples_acum[:, cl] * predictions_left

            for cl in range(pos_class_left+1, pos_class_right+1):
                probs_samples_prev_right[:, cl] = probs_samples_acum[:, cl] * predictions_right

            #aumular las probabiliades de cada rama
            self.__traverse_tree_ovo( X, probs_samples, probs_samples_prev_left, pos_class_left, pos_class_right-1, level+1)
            self.__traverse_tree_ovo(X, probs_samples, probs_samples_prev_right, pos_class_left+1, pos_class_right, level+1)


    def _predict_ovo(self, X):
        """
            Partiendo de la raiz del arbol, bajo por las ramas del mismo EJEMPLO a EJEMPLO
        :param X:
        :return:
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_samples_ini = np.ones((n_samples, n_classes))
        probs_samples = np.zeros((n_samples, n_classes))
        self.__traverse_tree_ovo(X, probs_samples, probs_samples_ini, 0, n_classes-1, level=0)
        return np.mean(probs_samples, axis=0)


    def _predict_ovo_freq(self, X):
        """
          Igual que el anterior pero contando las clases con probabilidades mayores (deberia ir peor)
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_samples_ini = np.ones((n_samples, n_classes))
        probs_samples = np.zeros((n_samples, n_classes))
        self.__traverse_tree_ovo(X, probs_samples, probs_samples_ini, 0, n_classes-1, level=0)

        classes_samples = np.argmax(probs_samples, axis=1)
        unique_elements, counts_elements = np.unique(classes_samples, return_counts=True)
        # Por si no todas las clases estan representadas
        cont = np.zeros(len(self.classes_))
        cont[unique_elements] = counts_elements
        return cont / float(np.sum(cont))  # frecuencias relativas



    def __obtain_leaves_ovo(self, X, cont_classes, pos_class_left, pos_class_right, level):
        #Recibe los ejemplos y cuenta las ramas (clases) en las que cae cada ejemplo

        n_classes = len(self.classes_)
        n_samples= X.shape[0]
        n_estim = (n_classes * n_classes - 1) / 2
        class_left = self.classes_[pos_class_left]
        class_right = self.classes_[pos_class_right]

        #caso base (hojas)
        if (pos_class_left-pos_class_right==0):
            cont_classes[pos_class_left]+=X.shape[0]
        elif X.shape[0] == 0:
            pass
        else:
            #obtengo el estimador de las dos clases consideradas
            i=0
            while i<n_estim and not (class_left==self.oqn_estimators_[i][1] and  class_right==self.oqn_estimators_[i][2]):
                i+=1
            estimador=self.oqn_estimators_[i][0]

            # ojo: pred son pares de la forma [prob_fallar prob_acertar]  -> prob_left=prob_acertar
            # obtengo las predicciones para cada ejemplo
            # predictions = estimador._predictions(X)  #version anterior -> mejor no meter nuevos metodos en los cuantificadores

            # En la nueva version de Alberto ya no hay quatificadores binarios especificos.
            # Son un caso particular de los multiclase, que tienen un diccionario de clasificadores binarios (one vs all)
            # Yo me quedo con el primero del diccionario porque se que es binario y solo hay uno
            predictions = estimador.estimators_[1].predict_proba(X)

            predictions_left = predictions[:, 1]
            #predictions_right = predictions[:, 0]

            samples_left = np.flatnonzero(predictions_left>= 0.5)
            samples_right = np.flatnonzero(predictions_left< 0.5)

            X_left = X[samples_left]
            X_right = X[samples_right]

            self.__obtain_leaves_ovo(X_left, cont_classes, pos_class_left, pos_class_right-1, level+1)
            self.__obtain_leaves_ovo(X_right, cont_classes, pos_class_left+1, pos_class_right ,level+1)

    def _predict_ovo_hojas(self, X):
        """
        Partiendo de la raiz del arbol, bajo por las ramas del mismo EJEMPLO a EJEMPLO
        y cada ejemplo se clasifica con la hoja por la que se llega
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        cont_classes = np.zeros(n_classes)
        self.__obtain_leaves_ovo(X, cont_classes, 0, n_classes-1, level=0)
        return cont_classes / float(np.sum(cont_classes))  # frecuencias relativas


    def __traverse_preleaves_ovo(self, X, mask_samples, probs_samples, pos_class_left, pos_class_right, level):
        # Recibe los ejemplos y baja hasta el nodo anterior a las ramas
        # voy enmascarando los ejemplos validos en cada rama por la que voy
        n_classes = len(self.classes_)
        n_samples= X.shape[0]
        n_estim = (n_classes * n_classes - 1) / 2
        class_left = self.classes_[pos_class_left]
        class_right = self.classes_[pos_class_right]

        #obtengo el estimador de las dos clases consideradas
        i=0
        while i<n_estim and not (class_left==self.oqn_estimators_[i][1] and  class_right==self.oqn_estimators_[i][2]):
            i+=1
        estimador=self.oqn_estimators_[i][0]

        # ojo: pred son pares de la forma [prob_fallar prob_acertar]  -> prob_left=prob_acertar
        #predictions = estimador._predictions(X)  # predicciones para cada ejemplo --> VIEJUNO
        predictions = estimador.estimators_[1].predict_proba(X)
        predictions_left = predictions[:, 1]
        predictions_right = predictions[:, 0]

        # Aplico la mascara para quedarme solo con los ejemplos de esta rama
        predictions_left[~mask_samples] = -1
        predictions_right[~mask_samples] = -1
        samples_left = np.flatnonzero(predictions_left >= 0.5)
        samples_right = np.flatnonzero(predictions_right > 0.5)

        #X_left = X[samples_left]
        #X_right = X[samples_right]


        mask_samples_left = deepcopy(mask_samples)
        mask_samples_right = deepcopy(mask_samples)
        mask_samples_left[samples_right] = False
        mask_samples_right[samples_left] = False

        # caso base  (nodo antes de las hojas: discrimina entre dos clases)
        # pongo la probabilidad de las dos clases implicadas y las otras a cero
        if (abs(pos_class_left - pos_class_right) == 1):
            if len(samples_left)>0:
                probs_samples[samples_left,pos_class_left]= predictions_left[samples_left]
                probs_samples[samples_left, pos_class_right]= predictions_right[samples_left]
            if len(samples_right)>0:
                probs_samples[samples_right,pos_class_left]= predictions_left[samples_right]
                probs_samples[samples_right, pos_class_right]= predictions_right[samples_right]

        else:
            self.__traverse_preleaves_ovo(X, mask_samples_left, probs_samples, pos_class_left, pos_class_right-1, level+1)
            self.__traverse_preleaves_ovo(X, mask_samples_right, probs_samples, pos_class_left+1, pos_class_right ,level+1)




    def _predict_ovo_hojas_prob(self, X):
        """
        Partiendo de la raiz del arbol, bajo por las ramas del mismo EJEMPLO a EJEMPLO hasta junsto antes de las hojas
        Para cada ejemplo, se asignan las probabilidades de las dos clases que discrimina el nodo (el resto cero????)
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_samples = np.zeros((n_samples, n_classes))
        self.__traverse_preleaves_ovo(X, np.ones(n_samples, dtype=np.bool), probs_samples, 0, n_classes-1, level=0)
        return np.mean(probs_samples, axis=0)


    def _predictions_ovo(self, X):
        """
            Partiendo de la raiz del arbol, bajo por las ramas del mismo EJEMPLO a EJEMPLO
            Retorna las probabilidades de cada ejemplo por si se quiere tener todo el detalle
        """
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        probs_samples_ini = np.ones((n_samples, n_classes))
        probs_samples = np.zeros((n_samples, n_classes))
        self.__traverse_tree_ovo(X, probs_samples, probs_samples_ini, 0, n_classes-1, level=0)
        return probs_samples