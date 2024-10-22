import ROOT
import matplotlib.pyplot as plt

# Path to your ROOT file
root_file_path = "root_files/full_op_ZH_M9_Alpha4_13.root"

# Open the ROOT file
file = ROOT.TFile(root_file_path, "READ")

# Access the tree in the ROOT file (Assuming it's named "Delphes" as in typical Delphes output)
tree = file.Get("Delphes")

# Create a histogram to store the PT of photons
hist_pt = ROOT.TH1F("hist_pt", "Photon PT;PT (GeV/c);Counts", 100, 0, 200)

# Loop over the entries in the tree and fill the histogram
for event in tree:
    for photon in event.Photon:
        hist_pt.Fill(photon.PT)

# Convert the ROOT histogram to a numpy array for plotting
n_bins = hist_pt.GetNbinsX()
x = [hist_pt.GetBinCenter(i+1) for i in range(n_bins)]
y = [hist_pt.GetBinContent(i+1) for i in range(n_bins)]

# Plot using matplotlib
plt.figure()
plt.plot(x, y, drawstyle='steps-mid')
plt.xlabel("PT (GeV/c)")
plt.ylabel("Counts")
plt.title("Transverse Momentum (PT) of Photons")
plt.grid(True)
plt.show()
