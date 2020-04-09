pipeline {
    options {
        disableConcurrentBuilds()
        timeout(time: 1, unit: 'HOURS')
    }
    agent { label "docker" }
    stages {
        stage ("build") {
            steps {
                sh "/usr/bin/git fetch --tags"
                sh "/usr/bin/docker build ."
            }
        }
    }
    post {
        cleanup {
            deleteDir()
        }
    }
}
