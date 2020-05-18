from gym.envs.registration import register

register(
    id='SimplePointBot-v0',
    entry_point='env.simplepointbot0:SimplePointBot'
)

register(
    id='SimplePointBot-v1',
    entry_point='env.simplepointbot1:SimplePointBot'
)

register(
    id='Maze-v0',
    entry_point='env.maze:MazeNavigation'
)

register(
    id='CliffWalker-v0',
    entry_point='env.cliffwalker:CliffWalkerEnv'
)

register(
    id='CliffCheetah-v0',
    entry_point='env.cliffcheetah:CliffCheetahEnv'
)

register(
    id='Shelf-v0',
    entry_point='env.shelf_env:ShelfEnv'
)

register(
    id='ShelfDynamic-v0',
    entry_point='env.shelf_dynamic_env:ShelfDynamicEnv'
)

register(
    id='CliffPusher-v0',
    entry_point='env.cliffpusher:PusherEnv'
)

register(
    id='Reacher-v0',
    entry_point='env.reacher:ReacherSparse3DEnv')

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

