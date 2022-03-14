import pygame
import os
import neat
from sys import exit
from random import randint

def displayScore(startTime):
    score = int(pygame.time.get_ticks()/1000)-startTime
    return score

def spawnObstacle(arraySnailR,obstacleTimer):
    for i in range(4):
        if(arraySnailR[i].right<0 or arraySnailR[i].left>1100):
            arraySnailR[i].left=800
            break 
    pygame.time.set_timer(obstacleTimer,randint(2000,2100))

def main(genomes,config):  
    pygame.init()
    screen = pygame.display.set_mode((800,400))
    pygame.display.set_caption('Runner NEAT')
    clock = pygame.time.Clock()
    font= pygame.font.Font("assets/font/Pixeltype.ttf",50)
    startTime=int(pygame.time.get_ticks()/1000)

    skyS = pygame.image.load('assets/graphics/Sky.png').convert()

    groundS = pygame.image.load('assets/graphics/ground.png').convert()
    groundR= groundS.get_rect(topleft=(0,300))

    # Timer spawn enymies
    obstacleTimer = pygame.USEREVENT+1

    arraySnailS = []
    arraySnailR = []
    animationSnail = []
    snailIndex= 0
    snailVelocity=6
    for i in range(4):
        snailSurf = pygame.image.load('assets/graphics/snail/snail1.png').convert_alpha()
        animationSnail.append([snailSurf,pygame.image.load('assets/graphics/snail/snail2.png').convert_alpha()])
        arraySnailS.append(snailSurf)
        arraySnailR.append(arraySnailS[i].get_rect(bottomright=(50000,300)))
    spawnObstacle(arraySnailR,obstacleTimer)

    nets=[]
    ge=[]
    runners = []
    animation = []

    for _,genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        nets.append(net)
        runnerSurf1 = pygame.image.load('assets/graphics/Player/player_walk_1.png').convert_alpha()
        runnerSurf2 = pygame.image.load('assets/graphics/Player/player_walk_2.png').convert_alpha()
        runnerSurfJump = pygame.image.load('assets/graphics/Player/jump.png').convert_alpha()
        runners.append([runnerSurf1,runnerSurf1.get_rect(midbottom=(80,301)),0,0])
        animation.append([runnerSurf1,runnerSurf2,runnerSurfJump])
        genome.fitness = 0
        ge.append(genome)

    running=True
    while running:
        clock.tick(60) #FPS of 60

        #Check if the user clicked the quit button and spawn obstacle
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(event)
                pygame.quit()
                exit()
            if event.type == obstacleTimer:
                spawnObstacle(arraySnailR,obstacleTimer)

        #draw the enviroment
        screen.blit(skyS,((0,0)))
        screen.blit(groundS,groundR)

        #increse the snail speed in time
        score = displayScore(startTime)
        if(score<1): score = 1
        scoreS=font.render(f'{score}',False,'Black')
        scoreR= scoreS.get_rect(center = (400,50))
        screen.blit(scoreS,scoreR)
        if(score%10==0): snailVelocity+=0.015

        for x,runner in enumerate(runners):
            #calculate the closer snail
            closerDistance = abs(arraySnailR[0].x-runner[1].x)
            for snail in arraySnailR:
                if snail.x-runner[1].x<closerDistance and snail.x-runner[1].x>0:
                    closerDistance=snail.x-runner[1].x
            
            output = nets[x].activate([closerDistance])
        #draw the runners
            screen.blit(runner[0],runner[1])

            #player input and move
            if(runner[1].y>=218):

                runner[0] = animation[x][int(runner[3])]
                runner[3]+=0.1
                if(runner[3]>=1.9):   runner[3]=0 
                if output[0]<-0.5:
                    runner[2]=-16 # JUMP
                    runner[1].y=215
            else:
                runner[0] = animation[x][2] 
                runner[1].y += runner[2] 
                runner[2] += 0.65  
                if(runner[1].y>215):
                    runner[1].y=218
                    runner[2] = 0
                
            #Moving the snails
            for i in range(4):
                if(runner[1].colliderect(arraySnailR[i])):
                    try:
                        ge[x].fitness -= 1
                        runners.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                        if len(runners)==0: 
                            running=False
                            break
                    except:
                        print('erro de index')
            
        #giving fitness with score
        for g in ge:
            g.fitness += (score/100)
        #Moving the snails
        snailIndex+=0.1
        if(snailIndex>=1.9):   snailIndex=0 
        for i in range(4):
            arraySnailS[i] = animationSnail[i][int(snailIndex)]
            if(arraySnailR[i].right<0):
                arraySnailR[i].left=50000
                for g in ge:
                    g.fitness +=0.1
                continue
            arraySnailR[i].x-=snailVelocity
            screen.blit(arraySnailS[i],arraySnailR[i])
        pygame.display.update()

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main,500)

if __name__=='__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)
