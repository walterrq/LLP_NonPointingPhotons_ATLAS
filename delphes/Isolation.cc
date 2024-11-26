#include "modules/Isolation.h"

#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "classes/DelphesFormula.h"

#include "ExRootAnalysis/ExRootClassifier.h"
#include "ExRootAnalysis/ExRootFilter.h"
#include "ExRootAnalysis/ExRootResult.h"

#include "TDatabasePDG.h"
#include "TFormula.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TRandom3.h"
#include "TString.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

//------------------------------------------------------------------------------

// ExRootClassifier, ExRootFilter, and Isolation: Handle classifications, filtering, and processing of these particles.

// The constructor initializes various pointers like fClassifier, fFilter, 
// and input iterators (fItIsolationInputArray, etc.).

// fClassifier is created as an instance of IsolationClassifier, 
// which is used to classify objects based on their transverse momentum (PTMin).
class IsolationClassifier : public ExRootClassifier
{
public:
  IsolationClassifier() {}

  Int_t GetCategory(TObject *object);

  Double_t fPTMin;
};

//------------------------------------------------------------------------------
// Various parameters for isolation are loaded via functions like GetDouble and GetBool
Int_t IsolationClassifier::GetCategory(TObject *object)
{
  // Represents physical particles or objects with properties like momentum, charge, and isolation variables.
  Candidate *track = static_cast<Candidate *>(object);
  const TLorentzVector &momentum = track->Momentum;

  if(momentum.Pt() < fPTMin) return -1;

  return 0;
}

//------------------------------------------------------------------------------

Isolation::Isolation() :
  fClassifier(0), fFilter(0),
  fItIsolationInputArray(0), fItCandidateInputArray(0),
  fItRhoInputArray(0)
{
  fClassifier = new IsolationClassifier;
}

//------------------------------------------------------------------------------

Isolation::~Isolation()
{
}

//------------------------------------------------------------------------------

void Isolation::Init()
{
  const char *rhoInputArrayName;

  // fDeltaRMax (default: 0.5): Maximum cone radius for charged particles.
  fDeltaRMax = GetDouble("DeltaRMax", 0.5);
  // fDeltaRMax_neutral (default: 0.2): Maximum cone radius for neutral particles
  fDeltaRMax_neutral = GetDouble("DeltaRMax_neutral", 0.2); // Initialize neutral DeltaRMax

  // fPTRatioMax (default: 0.1): Maximum ratio of isolation energy to candidate transverse momentum.
  fPTRatioMax = GetDouble("PTRatioMax", 0.1);

  // fPTSumMax (default: 5.0): Maximum isolation energy sum allowed for the candidate.
  fPTSumMax = GetDouble("PTSumMax", 5.0);

  //fUsePTSum (default: false): If true, isolation uses fPTSumMax instead of ratios.
  fUsePTSum = GetBool("UsePTSum", false);

  //fUseRhoCorrection (default: true): If true, corrects for background energy density.
  fUseRhoCorrection = GetBool("UseRhoCorrection", true);

  //fDeltaRMin (default: 0.01): Minimum cone radius.
  fDeltaRMin = GetDouble("DeltaRMin", 0.01);
  fUseMiniCone = GetBool("UseMiniCone", false);

  //fActivateMuonIso (default: false): Activates a muon-specific isolation algorithm.
  fActivateMuonIso = GetBool("ActivateMuonIso", false); // Initialize activation flag

  //The transverse momentum cutoff for classification (fClassifier->fPTMin) is set to 0.5.
  fClassifier->fPTMin = GetDouble("PTMin", 0.5);

  // import input array(s)

  // Represents an array of particles/objects used for isolation.

  // fIsolationInputArray: Default: Delphes/partons.
  fIsolationInputArray = ImportArray(GetString("IsolationInputArray", "Delphes/partons"));
  fItIsolationInputArray = fIsolationInputArray->MakeIterator();
  fFilter = new ExRootFilter(fIsolationInputArray);

  // Array of candidates (e.g., jets, electrons) for which isolation will be calculated.
  // fCandidateInputArray: Default: Calorimeter/electrons.
  fCandidateInputArray = ImportArray(GetString("CandidateInputArray", "Calorimeter/electrons"));
  fItCandidateInputArray = fCandidateInputArray->MakeIterator();

  //fRhoInputArray: Optional array for rho correction.
  rhoInputArrayName = GetString("RhoInputArray", "");
  if(rhoInputArrayName[0] != '\0')
  {
    fRhoInputArray = ImportArray(rhoInputArrayName);
    fItRhoInputArray = fRhoInputArray->MakeIterator();
  }
  else
  {
    fRhoInputArray = 0;
  }

  // create output array
  // fOutputArray: Stores candidates passing isolation criteria. Default: electrons.
  fOutputArray = ExportArray(GetString("OutputArray", "electrons"));
}

//------------------------------------------------------------------------------

// fOutputArray: Stores candidates passing isolation criteria. Default: electrons.
void Isolation::Finish()
{
  if(fItRhoInputArray) delete fItRhoInputArray;
  if(fFilter) delete fFilter;
  if(fItCandidateInputArray) delete fItCandidateInputArray;
  if(fItIsolationInputArray) delete fItIsolationInputArray;
}

//------------------------------------------------------------------------------

void Isolation::Process()
{
  Candidate *candidate, *isolation, *object;
  TObjArray *isolationArray;
  Double_t sumChargedNoPU, sumChargedPU, sumNeutral, sumAllParticles;
  Double_t sumNeutralEt, sum_Muon_Iso, ratio_Muon_Iso, sum_Tracks; // Declare missing variables
  Double_t sumDBeta, ratioDBeta, sumRhoCorr, ratioRhoCorr, sum, ratio;
  Bool_t pass = kFALSE;
  Double_t eta = 0.0;
  Double_t rho = 0.0;

  // select isolation objects
  fFilter->Reset();
  isolationArray = fFilter->GetSubArray(fClassifier, 0);
  TIter itIsolationArray(isolationArray);

  // loop over all input muons
  fItCandidateInputArray->Reset();
  while((candidate = static_cast<Candidate *>(fItCandidateInputArray->Next())))
  { 
    //Purpose: Iterate through all candidates from fCandidateInputArray to calculate their isolation properties.
    // Action: Retrieves the next Candidate object using the iterator fItCandidateInputArray
    // Example
    // Candidate: {Momentum: (Pt: 10.0, Eta: 0.0, Phi: 0.0), UniqueID: 1001}

    const TLorentzVector &candidateMomentum = candidate->Momentum;
    eta = TMath::Abs(candidateMomentum.Eta());

    // find rho
    rho = 0.0;
    // If the fRhoInputArray exists, the code resets its iterator (fItRhoInputArray) 
    // and loops through fRhoInputArray to find the rho value that corresponds to the candidate's eta.
    if(fRhoInputArray)
    {
      fItRhoInputArray->Reset();
      while((object = static_cast<Candidate *>(fItRhoInputArray->Next())))
      {
        if(eta >= object->Edges[0] && eta < object->Edges[1])
        {
          rho = object->Momentum.Pt();
        }
      }
    }

    // loop over all input tracks

    // sumNeutral: Sum of transverse momenta (Pt) of neutral particles.
    sumNeutral = 0.0;
    // sumChargedNoPU: Sum of transverse momenta of charged particles not flagged as pile-up (IsRecoPU=false).
    sumChargedNoPU = 0.0;
    // sumChargedPU: Sum of transverse momenta of charged pile-up particles (IsRecoPU=true).
    sumChargedPU = 0.0;
    // sumAllParticles: Total sum of transverse momenta of all particles passing isolation criteria.
    sumAllParticles = 0.0;

    //sumNeutralEt: Sum of transverse energy (Et) of neutral particles.
    sumNeutralEt = 0.0; // Initialize sumNeutralEt
    
    sum_Tracks = 0.0;

    //  These variables will be populated in the next loop (over fIsolationInputArray)
    itIsolationArray.Reset();
    while((isolation = static_cast<Candidate *>(itIsolationArray.Next())))
    {
      const TLorentzVector &isolationMomentum = isolation->Momentum;

      if (isolation->Charge != 0) // Charged particles use default cone values
      {
        if (fUseMiniCone)
        {
          pass = candidateMomentum.DeltaR(isolationMomentum) <= fDeltaRMax && 
                 candidateMomentum.DeltaR(isolationMomentum) > fDeltaRMin;
        }
        else
        {
          pass = candidateMomentum.DeltaR(isolationMomentum) <= fDeltaRMax && 
                 candidate->GetUniqueID() != isolation->GetUniqueID();
        }
      }
      else // Neutral particles use fDeltaRMax_neutral
      {
        if (fUseMiniCone)
        {
          pass = candidateMomentum.DeltaR(isolationMomentum) <= fDeltaRMax_neutral && 
                 candidateMomentum.DeltaR(isolationMomentum) > fDeltaRMin;
        }
        else
        {
          pass = candidateMomentum.DeltaR(isolationMomentum) <= fDeltaRMax_neutral && 
                 candidate->GetUniqueID() != isolation->GetUniqueID();
        }
      }

      if(pass)
      {
        sumAllParticles += isolationMomentum.Pt();
        if(isolation->Charge != 0)
        { 

          sum_Tracks += isolationMomentum.Pt();
          
          if(isolation->IsRecoPU)
          {
            sumChargedPU += isolationMomentum.Pt();
          }
          else
          {
            sumChargedNoPU += isolationMomentum.Pt();
          }
        }
        else
        {
          sumNeutral += isolationMomentum.Pt();
          sumNeutralEt += isolationMomentum.Et();
        }
      }
    }

    if (fActivateMuonIso)
    {
      // sum_Muon_Iso = sumChargedNoPU + 0.4 * sumNeutralEt;
      //ratio_Muon_Iso = sumChargedNoPU / candidateMomentum.Pt() + 
      //                 0.4 * sumNeutralEt / candidateMomentum.Et();

      sum_Muon_Iso = sum_Tracks + 0.4 * sumNeutralEt;
      ratio_Muon_Iso = sum_Tracks / candidateMomentum.Pt() + 
                       0.4 * sumNeutralEt / candidateMomentum.Et();

      candidate->IsolationVar = 0; // consistencia con el otro caso, pero no se utiliza
      candidate->IsolationVarRhoCorr = ratio_Muon_Iso;
      //candidate->SumPtCharged = sumChargedNoPU;
      candidate->SumPtCharged = sum_Tracks;
      candidate->SumPtNeutral = sumNeutral;
      candidate->SumPtChargedPU = sumChargedPU;
      candidate->SumPt = sumAllParticles;

      if(fUsePTSum && sum_Muon_Iso > fPTSumMax) continue;
      if(!fUsePTSum && ratio_Muon_Iso > fPTRatioMax) continue;

      fOutputArray->Add(candidate);
    }
    else
    {
      sumDBeta = sumChargedNoPU + TMath::Max(sumNeutral - 0.5 * sumChargedPU, 0.0);
      sumRhoCorr = sumChargedNoPU + TMath::Max(sumNeutral - TMath::Max(rho, 0.0) * 
                                                fDeltaRMax * fDeltaRMax * TMath::Pi(), 0.0);
      ratioDBeta = sumDBeta / candidateMomentum.Pt();
      ratioRhoCorr = sumRhoCorr / candidateMomentum.Pt();

      candidate->IsolationVar = ratioDBeta;
      candidate->IsolationVarRhoCorr = ratioRhoCorr;
      candidate->SumPtCharged = sumChargedNoPU;
      candidate->SumPtNeutral = sumNeutral;
      candidate->SumPtChargedPU = sumChargedPU;
      candidate->SumPt = sumAllParticles;

      sum = fUseRhoCorrection ? sumRhoCorr : sumDBeta;
      if(fUsePTSum && sum > fPTSumMax) continue;

      ratio = fUseRhoCorrection ? ratioRhoCorr : ratioDBeta;
      if(!fUsePTSum && ratio > fPTRatioMax) continue;

      fOutputArray->Add(candidate);
    }
  }
}
