#ifndef Isolation_h
#define Isolation_h

/** \class Isolation
 *
 *  Sums transverse momenta of isolation objects (tracks, calorimeter towers, etc)
 *  within a DeltaR cone around a candidate and calculates fraction of this sum
 *  to the candidate's transverse momentum. outputs candidates that have
 *  the transverse momenta fraction within (PTRatioMin, PTRatioMax].
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve
 *
 */

#include "classes/DelphesModule.h"

class TObjArray;

class ExRootFilter;
class IsolationClassifier;

class Isolation: public DelphesModule
{
public:
  Isolation();
  ~Isolation();

  void Init();
  void Process();
  void Finish();

private:
  Double_t fDeltaRMax;

  Double_t fDeltaRMax_neutral; // New: Separate DeltaRMax for neutral particles

  Double_t fPTRatioMax;

  Double_t fPTSumMax;

  Double_t fDeltaRMin;

  Bool_t fUsePTSum;

  Bool_t fUseRhoCorrection;

  Bool_t fUseMiniCone;

  Bool_t fActivateMuonIso; // Activate muon isolation logic

  IsolationClassifier *fClassifier; //!

  ExRootFilter *fFilter;

  TIterator *fItIsolationInputArray; //!

  TIterator *fItCandidateInputArray; //!

  TIterator *fItRhoInputArray; //!

  const TObjArray *fIsolationInputArray; //!

  const TObjArray *fCandidateInputArray; //!

  const TObjArray *fRhoInputArray; //!

  TObjArray *fOutputArray; //!

  ClassDef(Isolation, 1)
};

#endif

