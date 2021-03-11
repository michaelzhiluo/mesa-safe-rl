from gym.envs.registration import register

register(
    id='SimplePointBot-v0', entry_point='env.simplepointbot0:SimplePointBot')

register(
    id='SimplePointBot-v1', entry_point='env.simplepointbot1:SimplePointBot')

register(id='Maze-v0', entry_point='env.maze:MazeNavigation')
register(id='Maze1-v0', entry_point='env.mazes:Maze1Navigation')
register(id='Maze2-v0', entry_point='env.mazes:Maze2Navigation')
register(id='Maze3-v0', entry_point='env.mazes:Maze3Navigation')
register(id='Maze4-v0', entry_point='env.mazes:Maze4Navigation')
register(id='Maze5-v0', entry_point='env.mazes:Maze5Navigation')
register(id='Maze6-v0', entry_point='env.mazes:Maze6Navigation')

register(id='ImageMaze-v0', entry_point='env.image_maze:MazeImageNavigation')

register(id='CliffWalker-v0', entry_point='env.cliffwalker:CliffWalkerEnv')

register(id='CliffCheetah-v0', entry_point='env.cliffcheetah:CliffCheetahEnv')

register(id='Shelf-v0', entry_point='env.shelf_env:ShelfEnv')

register(
    id='ShelfDynamic-v0', entry_point='env.shelf_dynamic_env:ShelfDynamicEnv')

register(id='ShelfLong-v0', entry_point='env.shelf_long_env:ShelfLongEnv')

register(
    id='ShelfDynamicLong-v0',
    entry_point='env.shelf_dynamic_long_env:ShelfDynamicLongEnv')

register(id='ShelfReach-v0', entry_point='env.shelf_reach_env:ShelfRotEnv')

register(id='CliffPusher-v0', entry_point='env.cliffpusher:PusherEnv')

register(id='Reacher-v0', entry_point='env.reacher:ReacherSparse3DEnv')

register(id='Car-v0', entry_point='env.car:DubinsCar')

register(id='DVRKReacher-v0', entry_point='env.dvrk_reacher:DVRK_Reacher')

register(id='Minitaur-v0', entry_point='env.minitaur:MinitaurGoalVelocityEnv')

# Mujoco Envs
register(id='CartPoleLength-v0', entry_point='env.cartpole:CartPoleEnv', max_episode_steps=200)


register(id='Push-v0', entry_point='env.push:FetchPushEnv', max_episode_steps=50)
#register(id='HalfCheetah-Disabled-v0', entry_point='env.half_cheetah_disabled:HalfCheetahEnv')
# register(
#     id='MBRLPusherSparse-v0',
#     entry_point='dmbrl.env.pushersparse:PusherSparseEnv'
# )

# register(
#     id='MBRL-PickAndPlace-v1',
#     entry_point='dmbrl.env.pick_and_place:FetchPickAndPlaceEnv'
# )

# register(
#     id='AutograspCartgripper-v0',
#     entry_point='dmbrl.env.cartgripper:AutograspCartgripperEnv'
# )

# register(
#     id='TallCartgripper-v0',
#     entry_point='dmbrl.env.tall_cartgripper:TallCartgripperEnv'
# )

# register(
#     id='CartgripperXZGrasp-v0',
#     entry_point='dmbrl.env.cartgripper_xz_grasp:CartgripperXZGrasp'
# )
