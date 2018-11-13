class CellType:
    WALL = "#"  # character for wall
    BARRIER = "x"  # character for barrier
    SPACE = " "  # character for space
    TRAP = "*"  # character for trap
    SPEED = "^"  # character for speed
    BOMB = "!"  # character for bomb
    ARMOR = "@"  # character for armor

    powerup_list = [TRAP, SPEED, BOMB, ARMOR]

    @staticmethod
    def is_powerup(cell):
        return cell in CellType.powerup_list


class PowerupType:
    SPEED = "SPEED"
    ARMOR = "ARMOR"
