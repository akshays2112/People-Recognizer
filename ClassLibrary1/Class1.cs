using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Jeeves
{
    public static class Globals
    {
        public static decimal LearningRateForNeurons;
    }

    public interface Neuron
    {
        public Guid Id;
        public decimal Threshold;
        public bool FiredValue;
    }

    public class RawInputLayerNeuron : Neuron
    {
        public Guid Id;
        public decimal Weight;
        public decimal Threshold;
        public bool FiredValue;
        public List<NeuralConnection> Outputs;
        decimal currentInput;

        public RawInputLayerNeuron(decimal weight, decimal threshold)
        {
            Id = new Guid();
            Weight = weight;
            Threshold = threshold;
        }

        public void CalculateFiredValue(decimal input)
        {
            currentInput = input;
            FiredValue = Weight * input > Threshold ? true : false;
            foreach (NeuralConnection o in Outputs)
            {
                o.currentInputValue = FiredValue;
            }
        }

        public void TrainNeuron(bool desiredFiredValue)
        {
            if (desiredFiredValue != FiredValue)
            {
                Weight *= desiredFiredValue ? 1 + Globals.LearningRateForNeurons : 1 - Globals.LearningRateForNeurons;
                Threshold += (Weight - Threshold) * Globals.LearningRateForNeurons;
            }
        }
    }

    public class HiddenLayerNeuron : Neuron
    {
        public Guid Id;
        public List<NeuralConnection> Inputs;
        public List<NeuralConnection> Outputs;
        public decimal Threshold;
        public bool FiredValue;

        public HiddenLayerNeuron(decimal threshold)
        {
            Id = new Guid();
            Threshold = threshold;
        }

        public void TrainNeuron(bool desiredFiredValue)
        {
            if (desiredFiredValue != FiredValue)
            {
                decimal sumInputs = 0;
                foreach (NeuralConnection nc in Inputs)
                {
                    sumInputs += nc.currentInputValue ? nc.Weight : 0;
                }
                Threshold += (sumInputs - Threshold) * Globals.LearningRateForNeurons;
                foreach (NeuralConnection nc in Inputs)
                {
                    nc.Weight = nc.currentInputValue ? ((nc.Weight / sumInputs) * (Threshold - sumInputs) * Globals.LearningRateForNeurons) : 0;
                }
            }
        }

        public void CalculateFiredValue()
        {
            decimal sum = 0;
            foreach (NeuralConnection n in Inputs)
            {
                sum += n.currentInputValue ? n.Weight : 0;
            }
            FiredValue = sum > Threshold ? true : false;
            foreach (NeuralConnection o in Outputs)
            {
                o.currentInputValue = FiredValue;
            }
        }
    }

    public class RawOutputLayerNeuron : Neuron
    {
        public Guid Id;
        public List<NeuralConnection> Inputs;
        public decimal Threshold;
        public bool FiredValue;

        public RawOutputLayerNeuron(decimal threshold)
        {
            Id = new Guid();
            threshold = Threshold;
        }

        public void TrainNeuron(bool desiredFiredValue)
        {
            if (desiredFiredValue != FiredValue)
            {
                decimal sumInputs = 0;
                foreach (NeuralConnection nc in Inputs)
                {
                    sumInputs += nc.currentInputValue ? nc.Weight : 0;
                }
                Threshold += (sumInputs - Threshold) * Globals.LearningRateForNeurons;
                foreach (NeuralConnection nc in Inputs)
                {
                    nc.Weight = nc.currentInputValue ? ((nc.Weight / sumInputs) * (Threshold - sumInputs) * Globals.LearningRateForNeurons) : 0;
                }
            }
        }

        public void CalculateFiredValue()
        {
            decimal sum = 0;
            foreach (NeuralConnection n in Inputs)
            {
                sum += n.currentInputValue ? n.Weight : 0;
            }
            FiredValue = sum > Threshold ? true : false;
        }
    }

    public class NeuralConnection
    {
        public Guid FromNeuronId;
        public Object FromNeuron;
        public Guid ToNeuronId;
        public Object ToNeuron;
        public decimal Weight;
        public bool currentInputValue;

        public NeuralConnection(object fromNeuron, object toNeuron, decimal weight)
        {
            FromNeuron = fromNeuron;
            FromNeuronId = ((Neuron)fromNeuron).Id;
            ToNeuron = toNeuron;
            ToNeuronId = ((Neuron)toNeuron).Id;
            Weight = weight;
        }
    }

    public class NeuralLayer
    {
        public Guid Id;
        public List<object> LayerNeurons;

        public void AddNeuronToLayer(RawInputLayerNeuron riln)
        {
            LayerNeurons.Add((object)riln);
        }

        public void AddNeuronToLayer(HiddenLayerNeuron hln)
        {
            LayerNeurons.Add((object)hln);
        }

        public void AddNeuronToLayer(RawOutputLayerNeuron roln)
        {
            LayerNeurons.Add((object)roln);
        }
    }

    public class RecogAkshaysPictureNetwork
    {
        public List<NeuralLayer> Layers;

        public RecogAkshaysPictureNetwork()
        {
            Globals.LearningRateForNeurons = (decimal)0.00000000000000000043368086899420177360298112034798;
            NeuralLayer rawInputLayer = new NeuralLayer();
            for (int i = 0; i < 1024 * 1024; i++)
            {
                RawInputLayerNeuron riln = new RawInputLayerNeuron((decimal)0, (decimal)2097152);
                rawInputLayer.AddNeuronToLayer(riln);
            }
            Layers.Add(rawInputLayer);
            NeuralLayer hiddenLayer = new NeuralLayer();
            for (int i = 0; i < 1024 * 1024; i++)
            {
                HiddenLayerNeuron hln = new HiddenLayerNeuron((decimal)2199023255552);
                hiddenLayer.AddNeuronToLayer(hln);
            }
            Layers.Add(hiddenLayer);
            foreach (HiddenLayerNeuron hln in hiddenLayer.LayerNeurons)
            {
                foreach (RawInputLayerNeuron riln in rawInputLayer.LayerNeurons)
                {
                    NeuralConnection nc = new NeuralConnection(riln, hln, (decimal)2097152);
                    hln.Inputs.Add(nc);
                    riln.Outputs.Add(nc);
                }
            }
            NeuralLayer outputLayer = new NeuralLayer();
            RawOutputLayerNeuron roln = new RawOutputLayerNeuron((decimal)2305843009213693952);
            foreach (HiddenLayerNeuron hln in hiddenLayer.LayerNeurons)
            {
                NeuralConnection nc = new NeuralConnection(hln, roln, (decimal)2199023255552);
                hln.Outputs.Add(nc);
                roln.Inputs.Add(nc);
            }
        }

        public bool Predict(string filename)
        {
            Bitmap bmp = new Bitmap(filename);
            for (int y = 0; y < 1024; y++)
            {
                for (int x = 0; x < 1024; x++)
                {
                    for (int i = 0; i < Layers[0].LayerNeurons.Count; i++)
                    {
                        ((RawInputLayerNeuron)Layers[0].LayerNeurons[i]).CalculateFiredValue((decimal)bmp.GetPixel(x, y).ToArgb());
                    }
                }
                for (int i = 0; i < Layers[1].LayerNeurons.Count; i++)
                {
                    ((HiddenLayerNeuron)Layers[1].LayerNeurons[i]).CalculateFiredValue();
                }
                ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).CalculateFiredValue();
            }
            return ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).FiredValue;
        }

        public string TrainNetwork(string PicsFilePath, bool DesiredValue)
        {
            if (Directory.Exists(PicsFilePath))
            {
                string[] files = Directory.GetFiles(PicsFilePath);
                foreach (string file in files)
                {
                    Bitmap bmp = new Bitmap(file);
                    for (int y = 0; y < 1024; y++)
                    {
                        for (int x = 0; x < 1024; x++)
                        {
                            for (int i = 0; i < Layers[0].LayerNeurons.Count; i++)
                            {
                                ((RawInputLayerNeuron)Layers[0].LayerNeurons[i]).CalculateFiredValue((decimal)bmp.GetPixel(x, y).ToArgb());
                            }
                        }
                        for (int i = 0; i < Layers[1].LayerNeurons.Count; i++)
                        {
                            ((HiddenLayerNeuron)Layers[1].LayerNeurons[i]).CalculateFiredValue();
                        }
                        ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).CalculateFiredValue();
                        while (((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).FiredValue != DesiredValue)
                        {
                            ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).TrainNeuron(DesiredValue);
                            foreach (NeuralConnection nc in ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).Inputs)
                            {
                                ((HiddenLayerNeuron)nc.FromNeuron).TrainNeuron(DesiredValue);
                                foreach (NeuralConnection nc2 in ((HiddenLayerNeuron)nc.FromNeuron).Inputs)
                                {
                                    ((RawInputLayerNeuron)nc2.FromNeuron).TrainNeuron(DesiredValue);
                                }
                            }
                            for (int y2 = 0; y2 < 1024; y2++)
                            {
                                for (int x2 = 0; x2 < 1024; x2++)
                                {
                                    for (int i2 = 0; i2 < Layers[0].LayerNeurons.Count; i2++)
                                    {
                                        ((RawInputLayerNeuron)Layers[0].LayerNeurons[i2]).CalculateFiredValue((decimal)bmp.GetPixel(x2, y2).ToArgb());
                                    }
                                }
                                for (int i2 = 0; i2 < Layers[1].LayerNeurons.Count; i2++)
                                {
                                    ((HiddenLayerNeuron)Layers[1].LayerNeurons[i2]).CalculateFiredValue();
                                }
                                ((RawOutputLayerNeuron)Layers[2].LayerNeurons[0]).CalculateFiredValue();
                            }
                        }
                    }
                }
                return null;
            }
            else
            {
                return "Directory " + PicsFilePath + " does not exist.";
            }
        }

        public void SaveToFile(string filename)
        {
            Stream stream = File.Open(filename, FileMode.Create);
            BinaryFormatter bFormatter = new BinaryFormatter();
            bFormatter.Serialize(stream, Layers);
            stream.Close();
        }

        public void LoadFromFile(string filename)
        {
            Stream stream = File.Open(filename, FileMode.Open);
            BinaryFormatter bFormatter = new BinaryFormatter();
            Layers = (List<NeuralLayer>)bFormatter.Deserialize(stream);
            stream.Close();
        }
    }
}
