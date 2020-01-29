pipeline {
    options {
        disableConcurrentBuilds()
        timeout(time: 1, unit: 'HOURS')
    }
    agent { label "docker" }
    stages {
        stage ("notify-start") {
            steps {
                this.notifyBB("INPROGRESS")
            }
        }
        stage ("build") {
            steps {
                sh "/usr/bin/git fetch --tags"
                sh "/usr/bin/docker build ."
            }
        }
    }
    post {
        always {
            this.notifyBB(currentBuild.result)
        }
        cleanup {
            deleteDir()
        }
    }
}

def notifyBB(String state) {
    // on success, result is null
    state = state ?: "SUCCESS"

    if (state == "SUCCESS" || state == "FAILURE") {
        currentBuild.result = state
    }

    notifyBitbucket commitSha1: "${GIT_COMMIT}",
    credentialsId: 'aea50792-dda8-40e4-a683-79e8c83e72a6',
    disableInprogressNotification: false,
    considerUnstableAsSuccess: true,
    ignoreUnverifiedSSLPeer: false,
    includeBuildNumberInKey: false,
    prependParentProjectKey: false,
    projectKey: 'SW',
    stashServerBaseUrl: 'https://aicsbitbucket.corp.alleninstitute.org'
}
