docker run -d -it -v /mnt/DataResearch/DataStageData/ed_provider_notes/:/data -v /home/gsilver1/data/QuickUMLS/data/:/data/UMLS ahc-nlpie-docker.artifactory.umn.edu/qumls:1 /home/QuickUMLS/run.sh

docker run -d -it -v /mnt/DataResearch/DataStageData/ed_provider_notes/:/data ahc-nlpie-docker.artifactory.umn.edu/b9:1 /usr/share/biomedicus/scripts/run_biomedicus.sh

docker run --env  umlsUser='horcle' --env umlsPass='nEj123456' -d -it -v /mnt/DataResearch/DataStageData/ed_provider_notes/:/data ahc-nlpie-docker.artifactory.umn.edu/clmp:4 /usr/share/clamp/scripts/run_clamp.sh

docker run -d -it -v /mnt/DataResearch/gsilver1/note_test/ed_provider_notes/:/data ahc-nlpie-docker.artifactory.umn.edu/mm:4 /usr/share/public_mm/scripts/run_metamap.sh

docker run --env  ctakes_umlsuser='horcle' --env ctakes_umlspw='nEj123456' -d -it -v /mnt/DataResearch/DataStageData/ed_provider_notes/:/data ahc-nlpie-docker.artifactory.umn.edu/ctks:1 /usr/share/ctakes/scripts/run_ctakes.sh