using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Extensions;
using NumSharp.Utilities;

    public class INLayer
    {
        public NDArray ActiveOutput;
        public NDArray Output;
        public NDArray Input;
        public virtual NDArray Emit(NDArray input)
        {
            return null;
        }

        public virtual NDArray BackPropagate(NDArray loss, double learnRate,double timeT)
        {
            return null;
        }

        public virtual NDArray BackDe(NDArray delta)
        {
            return null;
        }
    }

    public enum ActiveFunction
    {
        Logistic,
        None,
        Tanh,
        ReLu
    }

    public enum Optimizer
    {
        Adam,
        None
    }
    public class Linear : INLayer
    {
        private delegate NDArray Func(NDArray src);
        public NDArray Weights;
        public NDArray Biases;
        public NDArray Weights_M;
        public NDArray Weights_V;

        public NDArray Biases_M;
        public NDArray Biases_V;
        private Func ActiveFunc;
        private Func DeActiveFunc;

        public Optimizer Optimizer;

        public ActiveFunction ActiveFunction;
        private NDArray Tanh(NDArray nDArray)
        {
            return np.tanh(nDArray);
        }
        private NDArray DTanh(NDArray nDArray)
        {
            
            return 1-np.square(np.tanh(nDArray));
        }

        private NDArray ReLu(NDArray src)
        {
            return 0.5 * (src + np.positive(src));
        }
        private NDArray DReLu(NDArray src)
        {
            return 0.5 * np.divide(src, np.positive(src) + 1e-8) + 0.5;
        }

        private NDArray Logistic(NDArray src)
        {
            return np.power((1 + np.exp(0 - src)), -1);
        }

        private NDArray DeLogistic(NDArray src)
        {
            return np.multiply(Logistic(src), (1 - Logistic(src)));
        }

        private NDArray NoneAct(NDArray src)
        {
            return src.Clone();
        }

        private NDArray DeNoneAct(NDArray src)
        {
            return np.ones(src.shape);
        }
        public Linear(int inputSize, int outputSize, ActiveFunction activeFunction,Optimizer optimizer)
        {
            this.ActiveFunction = activeFunction;
            this.Weights = np.random.uniform(0, Math.Sqrt(6.0 / (double)(inputSize + outputSize)), new int[] { outputSize, inputSize });
            

            

            this.Biases = np.random.normal(0, 0.1, new int[] { outputSize, 1 });
            this.Optimizer = optimizer;
            switch (optimizer) {

                case Optimizer.Adam:
                    {
                        this.Weights_M = np.zeros(Weights.shape);
                        this.Weights_V = np.zeros(Weights.shape);
                        this.Biases_M=np.zeros(Biases.shape);
                        this.Biases_V=np.zeros(Biases.shape);
                        break;
                    }
            }
            switch (activeFunction)
            {
                case ActiveFunction.Tanh:
                    {
                        this.ActiveFunc = this.Tanh;
                        this.DeActiveFunc = this.DTanh;
                        this.Weights = np.random.normal(0, Math.Sqrt(2.0 / (double)(inputSize + outputSize)), new int[] { outputSize, inputSize });
                        this.Biases = np.random.normal(0, 0.1, new int[] { outputSize, 1 });
                        break;
                    }
                case ActiveFunction.Logistic:
                    {
                        this.ActiveFunc = this.Logistic;
                        this.DeActiveFunc = this.DeLogistic;
                        this.Weights = np.random.normal(0, 4*Math.Sqrt(2.0 / (double)(inputSize + outputSize)), new int[] { outputSize, inputSize });
                        this.Biases = np.random.normal(0, 0.1, new int[] { outputSize, 1 });
                        break;
                    }
                case ActiveFunction.ReLu:
                    {
                        this.ActiveFunc = this.ReLu;
                        this.DeActiveFunc = this.DReLu;
                        this.Weights = np.random.normal(0, Math.Sqrt(2.0 / (double)(inputSize )), new int[] { outputSize, inputSize });
                        this.Biases = np.random.normal(0, 0.1, new int[] { outputSize, 1 });
                        break;
                    }
                case ActiveFunction.None:
                    {

                        this.ActiveFunc = this.NoneAct;
                        this.DeActiveFunc = this.DeNoneAct;
                        this.Weights = np.random.normal(0, Math.Sqrt(2.0 / (double)(inputSize + outputSize)), new int[] { outputSize, inputSize });
                        this.Biases = np.random.normal(0, 0.1, new int[] { outputSize, 1 });
                        break;

                    }
                default: throw new ArgumentOutOfRangeException();
            }

        }

        public override NDArray Emit(NDArray input)
        {
            this.Input = input.Clone();
            
            Output = np.matmul(Weights, input) + Biases;
            ActiveOutput = this.ActiveFunc(Output);
            return ActiveOutput;
        }

        public override NDArray BackPropagate(NDArray loss, double learnRate,double timeT)
        {
            switch (Optimizer)
            {
                case Optimizer.Adam:
                    {
                        timeT = timeT + 1;
                        double adbeta1 = 0.8;
                        double adbeta2 = 0.999;
                        var delta = np.multiply(loss, this.DeActiveFunc(Output));
                        var BiasLoss = np.sum(delta, delta.ndim - 1, NPTypeCode.Double).reshape(-1, 1) / ((double)delta.shape[delta.ndim - 1]);
                        this.Biases_M = adbeta1 * Biases_M + (1 - adbeta1) * BiasLoss;
                        this.Biases_V = adbeta2 * Biases_V + (1 - adbeta2) * np.square(BiasLoss);
                        var BMhat = Biases_M / (1 - np.power(adbeta1, timeT));
                        var BVhat=Biases_V/(1 - np.power(adbeta2, timeT));

                        this.Biases -= np.divide(BMhat,(np.sqrt(BVhat,NPTypeCode.Double)+1e-8)) * learnRate;
                        var BkLoss = np.matmul(this.Weights.transpose(), delta);
                        var WeightLoss = np.matmul(delta, this.Input.transpose()) / ((double)delta.shape[delta.ndim - 1]);

                        this.Weights_M = adbeta1 * Weights_M + (1 - adbeta1) * WeightLoss ;
                        this.Weights_V  = adbeta2 * Weights_V  + (1 - adbeta2) * np.square(WeightLoss);
                        var WMhat = Weights_M / (1 - np.power(adbeta1, timeT));
                        var WVhat = Weights_V / (1 - np.power(adbeta2, timeT));

                        this.Weights -= np.divide(WMhat, (np.sqrt(WVhat, NPTypeCode.Double) + 1e-8)) * learnRate;
                        return BkLoss;
                    }
                default:
                    {
                        var delta = np.multiply(loss, this.DeActiveFunc(Output));
                        var BiasLoss = np.sum(delta, delta.ndim - 1, NPTypeCode.Double).reshape(-1, 1) / ((double)delta.shape[delta.ndim - 1]);
                        this.Biases -= BiasLoss * learnRate;
                        var BkLoss = np.matmul(this.Weights.transpose(), delta);
                        var WeightLoss = np.matmul(delta, this.Input.transpose()) / ((double)delta.shape[delta.ndim - 1]);
                        this.Weights -= WeightLoss * learnRate;
                        return BkLoss;
                    }
            }
            
        }

        public override NDArray BackDe(NDArray delta)
        {
            delta = np.multiply(delta, this.DeActiveFunc(Output));
            var BkDelta = np.matmul(this.Weights.transpose(), delta);
            return BkDelta;
        }

    }

    public enum LossFunction
    {
        MSE,
        MAE
    }
    public class NNetwork
    {
        private delegate NDArray LossFunc(NDArray output, NDArray expect);
        public List<INLayer> Layers;
        private (NDArray eps, NDArray miu) InputBound;
        private (double eps, double miu) ActiveIBound;
        private (NDArray eps, NDArray miu) OutputBound;
        private (double eps, double miu) ActiveOBound;

        private LossFunc lossFunc;
        public void AddLayer(INLayer iNLayer)
        {
            this.Layers.Add(iNLayer);
        }

        private NDArray LossInputTrans(NDArray src)
        {
            return np.true_divide(src, InputBound.miu) * ActiveIBound.miu;
        }
        private NDArray InputTrans(NDArray src)
        {
            return np.true_divide((src - InputBound.eps), InputBound.miu)*ActiveIBound.miu+ActiveIBound.eps;
        }
        private NDArray LossInputReverse(NDArray src)
        {
            return np.multiply(src / ActiveIBound.miu, InputBound.miu);
        }


        private NDArray InputReverse(NDArray src)
        {
            return np.multiply((src-ActiveIBound.eps)/ActiveIBound.miu, InputBound.miu) + InputBound.eps;
        }

        private NDArray LossOutputTrans(NDArray src)
        {
            return np.multiply(src / ActiveOBound.miu, OutputBound.miu);
        }
        private NDArray OutputTrans(NDArray src)
        {
            
            return np.multiply((src - ActiveOBound.eps) / ActiveOBound.miu, OutputBound.miu) + OutputBound.eps;
        }
        private NDArray LossOutputReverse(NDArray src)
        {
            return np.true_divide(src, OutputBound.miu)*ActiveOBound.miu;
        }
        private NDArray OutputReverse(NDArray src)
        {
            return np.true_divide((src - OutputBound.eps), OutputBound.miu)*ActiveOBound.miu + ActiveOBound.eps; 
        }
        /// <summary>
        /// Nx2,average then half boundary
        /// </summary>
        /// <param name="inputBoundary"></param>
        /// <param name="outputBoundary"></param>
        /// 

        private NDArray MAE(NDArray nDArray,NDArray nDArray1)
        {
            throw new NotImplementedException();
        }

        private NDArray MSE(NDArray nDArray, NDArray nDArray1)
        {
            return nDArray-nDArray1;
        }

        /// <summary>
        /// 输入输出向量的上下限大小
        /// 最后一层激活函数性质
        /// 损失函数
        /// </summary>
        /// <param name="inputBoundary"></param>
        /// <param name="outputBoundary"></param>
        /// <param name="finalActive"></param>
        /// <param name="lossFunction"></param>
        /// <exception cref="NotImplementedException"></exception>
        public NNetwork((NDArray low, NDArray high) inputBoundary, (NDArray low, NDArray high) outputBoundary, ActiveFunction finalActive, LossFunction lossFunction)
        {
            this.Layers = new List<INLayer>();
            double v1 = 0;
            double v2 = 1;
            switch (finalActive)
            {
                case ActiveFunction.Tanh:
                    {
                        v1 = 0;
                        v2 = 1;
                        break;
                    }
                case ActiveFunction.ReLu:
                {
                v1= 0.5;
                    v2 = 0.5;
                        break;
                }
                case ActiveFunction.Logistic:
                    {
                        v1 = 0.5;
                        v2 = 0.5;
                        break;
                    }
                case ActiveFunction.None:
                    {
                        v1 = 0;
                        v2 = 1;
                        break;
                    }
                default:
                    throw new NotImplementedException();
            }
            this.ActiveIBound = (0, 1);
            this.ActiveOBound=  (v1,v2);
            
            this.InputBound = (np.add(inputBoundary.low, inputBoundary.high) / 2.0, np.add((-1.0) * inputBoundary.low, inputBoundary.high) / 2.0 * 1.2);
            this.OutputBound = ((np.add(outputBoundary.low, outputBoundary.high) / 2.0),( np.add((-1.0) * outputBoundary.low, outputBoundary.high) / 2.0 * 1.2));
            switch (lossFunction)
            {
                case LossFunction.MAE:
                    {
                        this.lossFunc = this.MAE;
                        break;
                    }
                case LossFunction.MSE:
                    {
                        this.lossFunc = this.MSE;
                        break;
                    }

                default: break;
            }
        }

        public NDArray DeInput(NDArray Input)
        {
            var input = InputTrans(Input);
            NDArray CurrentOutPut = Emit(Input);
            double eps = 1e-6;
            var delta = np.array(new double[] { eps }).reshape(1, 1);
            delta = LossOutputReverse(delta);
            for(int i=Layers.Count-1; i>=0; i--)
            {
                delta=Layers[i].BackDe(delta);
            }
            var tmp = LossInputReverse(delta);
            return 1.0/eps*tmp;

        }

        public NDArray Train(NDArray Input, NDArray Expect, double learnRate,double timeT)
        {
            var input = InputTrans(Input);
            NDArray CurrentOutPut = Emit(Input);
            NDArray loss = this.LossOutputReverse(this.lossFunc(CurrentOutPut, Expect));
            var temp = loss.Clone();
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                temp = Layers[i].BackPropagate(temp, learnRate,(double)timeT);
            }
            NDArray Loss = LossOutputTrans(loss);
            return np.power(np.sum(np.multiply(Loss, Loss),Loss.ndim-1,NPTypeCode.Double)/(double)(Loss.shape[Loss.ndim-1]), 0.5);
        }

        public NDArray Emit(NDArray Input)
        {
            NDArray input = InputTrans(Input);
            foreach (var layer in Layers)
            {
                input = layer.Emit(input);
            }
            return OutputTrans(input);
        }

    }


