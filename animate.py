import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from matplotlib.lines import Line2D

def animate_tweets(labels, coordinates, metadata, mode, interval=50):
    tweet_data = []
    for i, (meta, label, coord) in enumerate(zip(metadata, labels, coordinates)):
        if not meta or 'created_at' not in meta or 'username' not in meta:
            continue
            
        try:
            #handle timestamp formats
            timestamp_str = meta['created_at'].replace('Z', '+00:00')
            timestamp = datetime.fromisoformat(timestamp_str)
            
            tweet_data.append({
                'index': i,
                'timestamp': timestamp, 
                'label': label,
                'coord': coord,
                'username': meta['username'],
                'text': meta.get('text', 'No text available')
            })
        except (ValueError, KeyError):
            continue
    
    if not tweet_data:
        raise ValueError("no valid tweet data with timestamps found")
    
    tweet_data.sort(key=lambda x: x['timestamp'])
    
    fig = plt.figure(figsize=(12, 8))
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
    else:
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.set_facecolor('none')
    
    show_tweets = False
    current_frame = [0]
    
    unique_users = list(set(meta['username'] for meta in metadata if meta))
    user_colours = {user: ["black", "blue", "red", "green", "purple"][i % 5] 
                   for i, user in enumerate(unique_users)}
    
    def on_key_press(event):
        if event.key == 't':
            nonlocal show_tweets
            show_tweets = not show_tweets
            animate_frame(current_frame[0])
            fig.canvas.draw()
    
    def animate_frame(frame):
        current_frame[0] = frame
        ax.clear()
        
        if mode == '3d':
            ax.grid(True)
        else:
            ax.grid(False)
            ax.set_facecolor('none')
        
        current_tweets = tweet_data[:frame + 1]
        if not current_tweets:
            return []
        
        user_data = {}
        for user in unique_users:
            user_data[user] = {'clustered': [], 'noise': [], 'labels': []}
        
        for tweet in current_tweets:
            user = tweet['username']
            coord = tweet['coord']
            label = tweet['label']
            
            if label >= 0:
                user_data[user]['clustered'].append(coord)
                user_data[user]['labels'].append(label)
            else:
                user_data[user]['noise'].append(coord)
        
        #plot points for each user
        for user in unique_users:
            edge_color = user_colours[user]
            
            if user_data[user]['clustered']:
                clustered_coords = np.array(user_data[user]['clustered'])
                clustered_labels = np.array(user_data[user]['labels'])
                
                if mode == '3d':
                    ax.scatter(clustered_coords[:, 0], clustered_coords[:, 1], clustered_coords[:, 2],
                              c=clustered_labels, s=30, cmap="tab10", edgecolors=edge_color, 
                              linewidth=1.5, alpha=0.8)
                else:
                    ax.scatter(clustered_coords[:, 0], clustered_coords[:, 1],
                              c=clustered_labels, s=10, cmap="tab10", edgecolors=edge_color,
                              linewidth=1.0, alpha=0.8)
            
            if user_data[user]['noise']:
                noise_coords = np.array(user_data[user]['noise'])
                
                if mode == '3d':
                    ax.scatter(noise_coords[:, 0], noise_coords[:, 1], noise_coords[:, 2],
                              color="gray", s=20, alpha=0.5, edgecolors=edge_color, linewidth=1.0)
                else:
                    ax.scatter(noise_coords[:, 0], noise_coords[:, 1],
                              color="gray", s=8, alpha=0.5, edgecolors=edge_color, linewidth=0.8)
        
        start_time = current_tweets[0]['timestamp'].strftime('%Y-%m-%d')
        end_time = current_tweets[-1]['timestamp'].strftime('%Y-%m-%d')
        
        if mode == '2d':
            ax.set_xlabel(f"Tweet Timeline: {start_time} to {end_time} | Frame {frame+1}/{len(tweet_data)}")
            ax.set_ylabel("Press 't' to toggle tweet text display")
        else:
            ax.set_xlabel(f"Tweet Timeline: {start_time} to {end_time} | Frame {frame+1}/{len(tweet_data)}")
            ax.set_ylabel("Press 't' to toggle tweet text display")
            ax.set_zlabel("UMAP-3")
        
        if show_tweets:
            for tweet in current_tweets:
                tweet_text = f"@{tweet['username']}: {tweet['text'][:50]}..."
                coord = tweet['coord']
                
                if mode == '3d':
                    ax.text(coord[0], coord[1], coord[2], tweet_text,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                           fontsize=7, ha='center')
                else:
                    ax.annotate(tweet_text, (coord[0], coord[1]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                               fontsize=7, ha='left')
        
        return []
    
    #create animation
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(tweet_data),
                                 interval=interval, blit=False, repeat=False)
    
    print(f"Animation created: {len(tweet_data)} tweets over {mode.upper()} visualization")
    print("Controls: Press 't' to toggle tweet text display, close window to stop")
    
    plt.tight_layout()
    plt.show()
    
    return ani
