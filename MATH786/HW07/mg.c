/**
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "marriage_game.h"

#define d printf("LINE: %d\n", __LINE__);

const int SS = STAY_SINGLE;

enum DapSuccess {
  SUCCESS = 0,
  FAILURE
};


typedef struct MarriageGame {
  size_t n_proposers;
  size_t n_receivers;
  int* proposer_prefs;
  int* receiver_prefs;
} MarriageGame;

typedef struct MarriageGameSolution {
  int* proposers;
  int* receivers;
} MarriageGameSolution;

typedef struct Proposer {
  int next_proposal;
  int status;
} Proposer;


int
main(char** argv, int argc) {

    // mg_game_test(MG_VERBOSE);

    size_t n_men = 6;
    size_t n_women = 7;

    uint8_t men_prefs[] = {
        1,  2,  4,  0, SS,  3,  5,  6,
        1,  5,  6,  2, SS,  0,  3,  4,
        1,  2,  0,  3,  4,  5,  6, SS,
        4,  0,  2,  3,  1, SS,  5,  6,
        5,  6,  3,  1,  0, SS,  2,  4,
        6,  5,  1,  0,  2, SS,  3,  4,
    };
    uint8_t women_prefs[] = {
        0,  2,  1,  3, SS,  4,  5,

        1,  2,  5,  4,  3,  0, SS,
        1,  4,  3,  0,  2, SS,  5,
        2,  1,  0,  5,  3, SS,  4,
        0,  1,  2,  3,  4,  5, SS,
        5,  1,  2,  0,  3,  4, SS,
        4,  3,  0,  2,  1,  5, SS,
    };

    #define MEN 1
    #if MEN
    printf("%s\n", "MEN PROPOSING");
    mg_game_t* game = mg_game_new(n_men, n_women, men_prefs, women_prefs);
    #else
    printf("%s\n", "WOMEN PROPOSING");
    mg_game_t* game = mg_game_new(n_women, n_men, women_prefs, men_prefs);
    #endif

    mg_game_solve(game);

    printf("PROPOSERS\n");
    for (int i = 0; i < game->n_prop; i++) {
        printf("%d -> %-4d", i + 1, game->prop_matching[i] + 1);
    }
    printf("\nRECEIVERS\n");
    for (int i = 0; i < game->n_recv; i++) {
        printf("%d -> %-4d", i + 1, game->recv_matching[i] + 1);
    }
    printf("\n");


    mg_game_destroy(&game);
}






/*
int
main(char** argv, int argc) {
  // const size_t n_men = 5;
  // const size_t n_women = 4;
  // int men_prefs[] = {
  //   1,3,0,2,4,
  //   2,3,1,0,4,
  //   2,4,1,3,0,
  //   3,2,1,4,0,
  //   1,2,3,4,0,
  // };
  // int women_prefs[] = {
  //   2,3,5,1,4,0,
  //   2,3,0,1,4,5,
  //   2,3,1,4,5,0,
  //   5,1,3,4,2,0,
  // };

  const size_t n_men = 2;
  const size_t n_women = 2;
  int men_prefs[] = {
    0,1,STAY_SINGLE,
    0,1,STAY_SINGLE,
  };
  int women_prefs[] = {
    1,0,STAY_SINGLE,
    0,1,STAY_SINGLE,
  };


  MarriageGame men_proposing_game = {
    .n_proposers = n_men,
    .n_receivers = n_women,
    .proposer_prefs = men_prefs,
    .receiver_prefs = women_prefs
  };

  int* solution[n_men + n_women];

  deferred_acceptance_procedure(solution, &men_proposing_game);

  return SUCCESS;
}

int
deferred_acceptance_procedure(int* solution, const MarriageGame* game) {
  int n_prop = game->n_proposers;
  int n_recv = game->n_receivers;


  Proposer* proposers = (Proposer*) calloc(n_prop, sizeof(Proposer));
  for (int i = 0; i < n_prop; i++) {
    proposers[i].next_proposal = 0;
    proposers[i].status = DAP_REJECTED;
  }

  int* tentative_acceptances = (int*) calloc(n_recv, sizeof(int));
  for (int j = 0; j < n_recv; j++) {
    tentative_acceptances[j] = STAY_SINGLE; // default to staying single 
  }


  // for (int i = 0; i < n_prop; i++) {
  //   for (int j = 0; j < n_recv + 1; j++) {
  //     printf("%3d", game->proposer_prefs[(i * (n_recv + 1)) + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // for (int j = 0; j < n_receivers; j++) {
  //   for (int i = 0; i < n_proposers + 1; i++) {
  //     printf("%3d", receiver_prefs[(j * (n_proposers + 1)) + i]);
  //   }
  //   printf("\n");
  // }


  write_dap_step(stdout, proposers, n_prop, tentative_acceptances, n_recv);

  int n_rejected = n_recv;
  while (n_rejected > 0) {
    n_rejected = 0;
    

    for (int i = 0; i < n_prop; i++) {
      printf("Proposer %d:\n", i);

      proposer_action(tentative_acceptances, proposers, i, game);
    
      // printf("%d\n", proposers[i].status);
    }


    for (int i = 0; i < n_prop; i++) {
      if (proposers[i].status == DAP_REJECTED) {
        n_rejected += 1;
      }
    }

    write_dap_step(stdout, proposers, n_prop, tentative_acceptances, n_recv);
  }







  return SUCCESS;
}


int
proposer_action(int* tentative_acceptances, Proposer* proposers, int i, const MarriageGame* game) {
  Proposer* proposer = &(proposers[i]);

  if (proposer->status == DAP_REJECTED) {
    int j = game->proposer_prefs[(i * (game->n_receivers + 1)) + proposer->next_proposal];

    printf("\tProposing to %d\n", j);

    int preference = game->proposer_prefs[(i * (game->n_receivers + 1)) + j];
    if (preference == STAY_SINGLE) {
      proposer->status = DAP_STAY_SINGLE;
    
    } else {
      int ta = tentative_acceptances[j];

      for (int k = 0; k < game->n_receivers + 1; k++) {
        int preference = game->receiver_prefs[(j * (game->n_receivers + 1)) + k];
        // new proposal outranks tentative
        if (preference == i) {
          printf("\t%d is rejects %d's proposal in favor of %d\n", j, ta, i);

          if (ta != STAY_SINGLE) {
            proposers[ta].status = DAP_REJECTED;
            proposers[ta].next_proposal += 1;
          }
          proposer->status = DAP_TENTATIVE_ACCEPTED;
          tentative_acceptances[j] = i;
          break;

        } else if (preference == ta) { // tentative outranks proposer
          printf("\t%d is keeps %d over %d\n", j, ta, i);
          proposer->status = DAP_REJECTED;
          proposer->next_proposal += 1;
          break;
        }
      }
    }
  } else {
    printf("\tProposer %d is not rejected currently.\n", i);
  }
}



int
write_dap_step(FILE* out, Proposer* proposers, size_t n_proposers, int* tentative_acceptances, size_t n_receivers) {
  fprintf(out, "proposers:\n");
  for (int i = 0; i < n_proposers; i++) {
    int status = proposers[i].status;
    char s[5];

    switch (status) {
      case DAP_REJECTED:
        strcpy(s, "REJ");
        break;

      case DAP_STAY_SINGLE:
        strcpy(s, "SS");
        break;

      case DAP_TENTATIVE_ACCEPTED:
        strcpy(s, "TEN");
        break;

      default:
        strcpy(s, "");
        break;
    }

    fprintf(out, "%5s  ", s);
  }
  fprintf(out, "\n");
  // for (int i = 0; i < n_proposers; i++) {
  //   fprintf(out, "%5d  ", proposers[i].next_proposal);
  // }
  // fprintf(out, "\n");

  fprintf(out, "Tentative Acceptances:\n");
  for (int j = 0; j < n_receivers; j++) {
    if (tentative_acceptances[j] != STAY_SINGLE) {
      fprintf(out, "%5d:%4d", j, tentative_acceptances[j]);
    } else {
      fprintf(out, "%5d:%4s", j, "SS" );
    }
  }
  fprintf(out, "\n");

  fprintf(out, "%s\n", "=============================\n");
}
*/
